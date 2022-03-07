#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/vfh.h>
#include <flann/flann.h>
#include <boost/filesystem.hpp>
#include <ctime>

typedef std::pair<boost::filesystem::path, std::vector<float>> vfh_model;

bool calculateVFH (const boost::filesystem::path &path, vfh_model &model)
{
  // Object for storing the point cloud.
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  // Object for storing the normals.
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	// Object for storing the VFH descriptor.
	pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptor(new pcl::PointCloud<pcl::VFHSignature308>);
  // Read a PLY file from disk.
	if (pcl::io::loadPLYFile<pcl::PointXYZ>(path.string(), *cloud) != 0)
	{
    pcl::console::print_info ("Wrong file name\n"); 
		return (false);
	}
  
  // Normals estimation.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
	normalEstimation.setInputCloud(cloud);
	normalEstimation.setRadiusSearch(0.03);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
  // VFH estimation.
	pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(cloud);
	vfh.setInputNormals(normals);
	vfh.setSearchMethod(kdtree);
  vfh.setNormalizeBins(true);
  vfh.setNormalizeDistance(true);
  vfh.compute(*descriptor);
  // Convert PCL data to a standard vector.
  pcl::PointCloud <pcl::VFHSignature308> pointCloud = *descriptor;
  model.second.resize(308);
  for (std::size_t i = 0; i < 308; ++i)
    model.second[i] = pointCloud[0].histogram[i];
    
  model.first = path;

  return (true);
}

void
loadData (const boost::filesystem::path &base_dir, const std::string &extension, std::vector<vfh_model> &models)
{
  if (!boost::filesystem::exists(base_dir) && !boost::filesystem::is_directory(base_dir))
    return;

  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator(); ++it)
  {
    if (boost::filesystem::is_directory(it->status()))
    {
      std::stringstream ss;
      ss << it->path();
      pcl::console::print_highlight ("Loading %s (%lu models loaded so far).\n", ss.str().c_str(), (unsigned long)models.size());
      loadData(it->path(), extension, models);
    }
    if (boost::filesystem::is_regular_file (it->status()) && boost::filesystem::extension (it->path()) == extension)
    {
      vfh_model m;
      if (calculateVFH(base_dir/it->path().filename(), m))
        models.push_back(m);
    }
  }
}

inline void
nearestKSearch(flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model, int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
  // Query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size()], 1, model.second.size ());
  memcpy (&p.ptr()[0], &model.second[0], p.cols * p.rows * sizeof(float));

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);
  index.knnSearch (p, indices, distances, k, flann::SearchParams(512));
  delete[] p.ptr();
}

void
normalize(std::vector<vfh_model> &training_models, std::vector<vfh_model> &test_models)
{ 
  std::vector<float> max(training_models[0].second.size());
  std::vector<float> min(training_models[0].second.size());
  for(int j=0; j<training_models[0].second.size(); ++j){
    max[j]=0.0001;
    min[j]=10000.0;
    for(int i=0; i<training_models.size(); ++i){
      if(max[j]<training_models[i].second[j])
        max[j]=training_models[i].second[j];
      if(min[j]>training_models[i].second[j])
        min[j]=training_models[i].second[j];
    }
    for(int i=0; i<test_models.size(); ++i){
      if(max[j]<test_models[i].second[j])
        max[j]=test_models[i].second[j];
      if(min[j]>test_models[i].second[j])
        min[j]=test_models[i].second[j];
    }
  }

  for(int i=0; i<training_models.size(); ++i){
    for(int j=0; j<training_models[0].second.size(); ++j){
      float temp=(training_models[i].second[j]-min[j])/(max[j]-min[j]);
      training_models[i].second[j]=temp;
    }
  }
  for(int i=0; i<test_models.size(); ++i){
    for(int j=0; j<training_models[0].second.size(); ++j){
      float temp=(test_models[i].second[j]-min[j])/(max[j]-min[j]);
      test_models[i].second[j]=temp;
    }
  }
}

void
calculateFeatureVector(std::vector<vfh_model> &models, std::vector<vfh_model> &features)
{
  // Calculate mean values.
  features.resize(models.size());
  for (int i = 0; i < models.size(); ++i){
    features[i].second.resize(10);
    float l(0.0);
    float m(0.0);
    for (int j = 0; j < 45; ++j){
      l=l+(models[i].second[j])*j;
      m=m+models[i].second[j];
    }
    features[i].second[0]=l/m;
    l=0;
    m=0;
    for (int j = 45; j < 90; ++j){
      l=l+(models[i].second[j])*j;
      m=m+models[i].second[j];
    }
    features[i].second[1]=l/m;
    l=0;
    m=0;
    for (int j = 90; j < 135; ++j){
      l=l+(models[i].second[j])*j;
      m=m+models[i].second[j];
    }
    features[i].second[2]=l/m;
    l=0;
    m=0.0001;
    for (int j = 135; j < 180; ++j){
      l=l+(models[i].second[j])*j;
      m=m+models[i].second[j];
    }
    features[i].second[3]=l/m;
    l=0;
    m=0;
    for (int j = 180; j < 308; ++j){
      l=l+(models[i].second[j])*j;
      m=m+models[i].second[j];
    }
    features[i].second[4]=l/m;
    l=0;
    m=0;
  
    features[i].first = models[i].first;
  }
  // Calculate standard deviation.
  for (int i = 0; i < models.size(); ++i){
    float l(0.0);
    float m(0.0);
    for (int j = 0; j < 45; ++j){
      l=l+(powf((j-features[i].second[0]),2))*models[i].second[j];
      m=m+models[i].second[j];
    }
    features[i].second[5]=sqrtf(l/m);
    l=0;
    m=0;
    for (int j = 45; j < 90; ++j){
      l=l+powf((j-features[i].second[1]),2)*models[i].second[j];
      m=m+models[i].second[j];
    }
    features[i].second[6]=sqrtf(l/m);
    l=0;
    m=0;
    for (int j = 90; j < 135; ++j){
      l=l+powf((j-features[i].second[2]),2)*models[i].second[j];
      m=m+models[i].second[j];
    }
    features[i].second[7]=sqrtf(l/m);
    l=0;
    m=0.0001;
    for (int j = 135; j < 180; ++j){
      l=l+powf((j-features[i].second[3]),2)*models[i].second[j];
      m=m+models[i].second[j];
    }
    features[i].second[8]=sqrtf(l/m);
    l=0;
    m=0;
    for (int j = 180; j < 308; ++j){
      l=l+powf((j-features[i].second[4]),2)*models[i].second[j];
      m=m+models[i].second[j];
    }
    features[i].second[9]=sqrtf(l/m);
    l=0;
    m=0;
  }
}

int
main(int argc, char** argv)
{
	bool use_vfh = false;
  // Object for storing the path to the training data.
  boost::filesystem::path training_set = "train2";
  
  // Object for storing the path to the test data.
  boost::filesystem::path test_set = "test2";
  std::string extension = ".ply";
  transform (extension.begin (), extension.end (), extension.begin (), (int(*)(int))tolower);


  // Calculate VFH descriptors for the training data.
  std::vector<vfh_model> training_models;
	loadData(training_set, extension, training_models);
  pcl::console::print_highlight ("Loaded %lu training models in total.\n", (unsigned long)training_models.size());
  
  // Calculate VFH descriptors for the test data.
  std::vector<vfh_model> test_models;
	loadData(test_set, extension, test_models);
  pcl::console::print_highlight ("Loaded %lu test models in total.\n", (unsigned long)test_models.size());
  
  // Calculate feature vectors for the training data.
  std::vector<vfh_model> training_features;
  calculateFeatureVector(training_models, training_features);
  
  // Calculate feature vectors for the test data.
  std::vector<vfh_model> test_features;
  calculateFeatureVector(test_models, test_features);
  
  if(use_vfh){
  training_models = training_features;
  test_models = test_features;
  }

  // Normalize feature vectors.
  normalize(training_models, test_models);

  // Convert data into FLANN format.
  flann::Matrix<float> training_data (new float[training_models.size() * training_models[0].second.size() ], training_models.size(), training_models[0].second.size());
  for (std::size_t i = 0; i < training_data.rows; ++i)
    for (std::size_t j = 0; j < training_data.cols; ++j)
      training_data[i][j] = training_models[i].second[j];
    
  // Build KNN index.
  flann::Index<flann::ChiSquareDistance<float> > index (training_data, flann::KDTreeIndexParams (4));
  index.buildIndex ();
 
  // KNN search.
  int k = 1;
  int type_count = 0;
  int total_type = 0;
  int total_count = 0;
  double thresh = DBL_MAX;   
  flann::Matrix<int> k_indices;
  flann::Matrix<float> k_distances;

  clock_t zegar = clock();
  for(int i = 0; i<test_models.size(); ++i){
    nearestKSearch (index, test_models[i], k, k_indices, k_distances);
    ++total_type;

    if(test_models[i].first.parent_path().leaf().string()==training_models.at(k_indices[0][0]).first.parent_path().leaf().string())
      ++type_count;
  
    if(i < test_models.size()-1 && test_models[i].first.parent_path().leaf().string()!=test_models[i+1].first.parent_path().leaf().string()){
      total_count+=type_count;
      //pcl::console::print_highlight ("Accuracy for %s: %d/%d %.2f% \n", test_models[i].first.parent_path().leaf().c_str(), type_count, total_type, (double(type_count)/double(total_type))*100);
      type_count = 0;
      total_type = 0;
    }
    if(i == test_models.size()-1){
      total_count+=type_count;
      //pcl::console::print_highlight ("Accuracy for %s: %d/%d %.2f% \n", test_models[i].first.parent_path().leaf().c_str(), type_count, total_type, (double(type_count)/double(total_type))*100);
    }  
 }
  
  //pcl::console::print_highlight ("Total accuracy: %d/%d %.2f% \n", total_count, test_models.size(), (double(total_count)/double(test_models.size()))*100);
  printf("Czas wykonywania: %lu ms\nŚredni czas jednego porównania: %lu ms \n", clock()-zegar, (clock()-zegar)/test_models.size());
  delete[] training_data.ptr();
  return (0);
}
