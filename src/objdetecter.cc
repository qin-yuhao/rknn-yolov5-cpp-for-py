
#include <stdio.h>
#include <sys/time.h>
#include <thread>
#include <queue>
#include <vector>
#define _BASETSD_H
#ifndef DEMO_PYBIND11_SRC_DEMO_H_
#define DEMO_PYBIND11_SRC_DEMO_H_
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Python.h> // Add the missing include for Python.h
namespace py = pybind11;
#endif // !DEMO_PYBIND11_SRC_DEMO_H_
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include "ThreadPool.hpp"
#include "mat_warper.h"
using std::queue;
using std::time;
using std::time_t;
using std::vector;




class Rknn_yolo_runner 
{
private:
  int init_count = 0;
  int init_flag = 0;
  int put_count = 0;
  std::string model_path;
  int thread_num;
  vector<rknn_lite *> rkpool;
  queue<std::future<std::vector<cv::Mat>>> futs;
  dpool::ThreadPool* pool;
  int frames = 0;
public:
  Rknn_yolo_runner(const std::string &model_path, int thread_num);
  ~Rknn_yolo_runner();
  py::list get_put(py::array_t<unsigned char>& input);
  void clean_pool();
  void init_pool(cv::Mat &input);
  void clean();
  //销毁线程池
  //void destroy_pool();
};

Rknn_yolo_runner::Rknn_yolo_runner(const std::string &model_path, int thread_num)
{
    this->model_path = model_path; 
    printf("model_path:%s\n", this->model_path.c_str ()); // 使用 c_str () 转换
    this->thread_num = thread_num;
    this->pool = new dpool::ThreadPool(thread_num);
}

Rknn_yolo_runner::~Rknn_yolo_runner()
{
    this->clean_pool();
}

void Rknn_yolo_runner::init_pool(cv::Mat &input)
{
  printf("init_pool\n");
  rknn_lite *ptr = new rknn_lite(const_cast<char*> (this->model_path.c_str ()), this->init_count %3); 
  this->rkpool.push_back(ptr);
  //ptr->ori_img = input;
  input.copyTo(ptr->ori_img);
  this->futs.push(this->pool->submit(&rknn_lite::interf, &(*ptr)));
}

py::list Rknn_yolo_runner::get_put(py::array_t<unsigned char>& input)
{
  py::list result;
  cv::Mat img = numpy_uint8_3c_to_cv_mat(input);
  if (this->init_flag == 0)
  {
    if (this->init_count < this->thread_num)
    {
      this->init_pool(img);
      this->init_count++;
      this->put_count++;
      printf("init_count:%d\n", this->init_count);
      if (this->init_count == this->thread_num)
          this->init_flag = 1;
      result.append(-1);
      return result;
    }
  }
  else if (this->init_flag == 1)
  {
    if (this->put_count < this->thread_num)//put 直到线程池满
      {
        img.copyTo(rkpool[frames % this->thread_num]->ori_img);
        futs.push(this->pool->submit(&rknn_lite::interf, &(*rkpool[frames++ % this->thread_num])));
        this->put_count++;
        result.append(-1);
        return result;
      }
    else
    {
      std::vector<cv::Mat> pred = futs.front().get();
      futs.pop();
      img.copyTo(rkpool[frames % this->thread_num]->ori_img);
      futs.push(this->pool->submit(&rknn_lite::interf, &(*rkpool[frames++ % this->thread_num])));

      if(!pred.empty())
      {
        for(int i = 0; i < pred.size(); i++)
        {
          //leteerbox
          int w = pred[i].cols;
          int h = pred[i].rows;
          if (w == 0 || h == 0)
          {
            printf("w or h == 0\n");
            continue;
          }
          py::array_t<unsigned char> tmp = cv_mat_uint8_3c_to_numpy(pred[i]);
          result.append(tmp);
        }
        return result;
      }
    }
  }
  return result;
}

void Rknn_yolo_runner::clean_pool()
{
  // 释放剩下的资源
  while (!futs.empty())
  {
    futs.pop();
  }
  for (int i = 0; i < this->thread_num; i++)
    delete rkpool[i];
}
void Rknn_yolo_runner::clean()
{
  // 释放剩下的资源
  while (!futs.empty())
  {
    futs.pop();
  }
  this->put_count = 0;
}
PYBIND11_MODULE(objdetecter, m) {
  m.doc() = "rknn cpp for python"; // optional module docstring
  py::class_<Rknn_yolo_runner>(m, "Rknn_yolo_runner")
    .def(py::init<char*, int>(), py::arg("model_path"), py::arg("thread_num"))
    .def("get_put", &Rknn_yolo_runner::get_put,py::return_value_policy::move, py::arg("input"))
    .def("clean_pool", &Rknn_yolo_runner::clean_pool)
    .def("init_pool", &Rknn_yolo_runner::init_pool)
    .def("clean", &Rknn_yolo_runner::clean);
}

