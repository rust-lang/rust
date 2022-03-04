#!/bin/bash

BENCHDIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# install dependencies
sudo apt update -y
sudo apt install gcc g++ gfortran cmake libboost-all-dev python autoconf libblas-dev -y

# Disable hyperthreading/etc
cd "$BENCHDIR"
sudo ../hyper.sh

# Build LLVM 8
git clone https://github.com/llvm/llvm-project -b release/8.x ~/llvm-project
cd ~/llvm-project
mkdir build
cd build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang" -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release
make -j16

# Build Enzyme
cd "$BENCHDIR"/..
mkdir build
cd build
cmake .. -DLLVM_DIR="$HOME/llvm-project/build/lib/cmake/llvm" -DLLVM_EXTERNAL_LIT="$HOME/llvm-project/build/bin/llvm-lit"
make -j16

make bench-ba bench-lstm bench-gmm bench-odeconst bench-ode bench-fft bench-odereal

cd "$BENCHDIR"
#echo "Pre optimization"
#./getdata.sh

for z in ba lstm gmm ode-const ode fft ode-real;
do
    cp $z/Makefile.makeafter $z/Makefile.make
done

cd "$BENCHDIR"/../build
make bench-ba bench-lstm bench-gmm bench-odeconst bench-ode bench-fft bench-odereal

#echo "Post optimization"
#cd "$BENCHDIR"
#./getdataafter.sh

