set -e

cd "build"
cmake j4  ..
cd ..
python setup.py build_ext --inplace