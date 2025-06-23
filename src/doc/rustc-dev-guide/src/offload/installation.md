# Installation

In the future, `std::offload` should become available in nightly builds for users. For now, everyone still needs to build rustc from source. 

## Build instructions

First you need to clone and configure the Rust repository:
```bash
git clone --depth=1 git@github.com:rust-lang/rust.git
cd rust
./configure --enable-llvm-link-shared --release-channel=nightly --enable-llvm-assertions --enable-offload --enable-enzyme --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs
```

Afterwards you can build rustc using:
```bash
./x.py build --stage 1 library
```

Afterwards rustc toolchain link will allow you to use it through cargo:
```
rustup toolchain link offload build/host/stage1
rustup toolchain install nightly # enables -Z unstable-options
```



## Build instruction for LLVM itself
```bash
git clone --depth=1 git@github.com:llvm/llvm-project.git 
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_TARGETS_TO_BUILD="host,AMDGPU,NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="offload,openmp" -DLLVM_ENABLE_PLUGINS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.
ninja
ninja install
```
This gives you a working LLVM build.


## Testing
run
```
./x.py test --stage 1 tests/codegen/gpu_offload
```

## Usage
It is important to use a clang compiler build on the same llvm as rustc. Just calling clang without the full path will likely use your system clang, which probably will be incompatible.
```
/absolute/path/to/rust/build/x86_64-unknown-linux-gnu/stage1/bin/rustc --edition=2024 --crate-type cdylib src/main.rs --emit=llvm-ir  -O -C lto=fat -Cpanic=abort -Zoffload=Enable
/absolute/path/to/rust/build/x86_64-unknown-linux-gnu/llvm/bin/clang++ -fopenmp --offload-arch=native -g  -O3 main.ll -o main -save-temps
LIBOMPTARGET_INFO=-1  ./main
```
The first step will generate a `main.ll` file, which has enough instructions to cause the offload runtime to move data to and from a gpu.
The second step will use clang as the compilation driver to compile our IR file down to a working binary. Only a very small Rust subset will work out of the box here, unless
you use features like build-std, which are not covered by this guide. Look at the codegen test to get a feeling for how to write a working example.
In the last step you can run your binary, if all went well you will see a data transfer being reported:
```
omptarget device 0 info: Entering OpenMP data region with being_mapper at unknown:0:0 with 1 arguments:
omptarget device 0 info: tofrom(unknown)[1024]
omptarget device 0 info: Creating new map entry with HstPtrBase=0x00007fffffff9540, HstPtrBegin=0x00007fffffff9540, TgtAllocBegin=0x0000155547200000, TgtPtrBegin=0x0000155547200000, Size=1024, DynRefCount=1, HoldRefCount=0, Name=unknown
omptarget device 0 info: Copying data from host to device, HstPtr=0x00007fffffff9540, TgtPtr=0x0000155547200000, Size=1024, Name=unknown
omptarget device 0 info: OpenMP Host-Device pointer mappings after block at unknown:0:0:
omptarget device 0 info: Host Ptr           Target Ptr         Size (B) DynRefCount HoldRefCount Declaration
omptarget device 0 info: 0x00007fffffff9540 0x0000155547200000 1024     1           0            unknown at unknown:0:0
// some other output
omptarget device 0 info: Exiting OpenMP data region with end_mapper at unknown:0:0 with 1 arguments:
omptarget device 0 info: tofrom(unknown)[1024]
omptarget device 0 info: Mapping exists with HstPtrBegin=0x00007fffffff9540, TgtPtrBegin=0x0000155547200000, Size=1024, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
omptarget device 0 info: Copying data from device to host, TgtPtr=0x0000155547200000, HstPtr=0x00007fffffff9540, Size=1024, Name=unknown
omptarget device 0 info: Removing map entry with HstPtrBegin=0x00007fffffff9540, TgtPtrBegin=0x0000155547200000, Size=1024, Name=unknown
```
