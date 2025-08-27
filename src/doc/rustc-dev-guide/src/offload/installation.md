# Installation

`std::offload` is partly available in nightly builds for users. For now, everyone however still needs to build rustc from source to use all features of it. 

## Build instructions

First you need to clone and configure the Rust repository:
```bash
git clone git@github.com:rust-lang/rust
cd rust
./configure --enable-llvm-link-shared --release-channel=nightly --enable-llvm-assertions --enable-llvm-offload --enable-llvm-enzyme --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs
```

Afterwards you can build rustc using:
```bash
./x build --stage 1 library
```

Afterwards rustc toolchain link will allow you to use it through cargo:
```
rustup toolchain link offload build/host/stage1
rustup toolchain install nightly # enables -Z unstable-options
```



## Build instruction for LLVM itself
```bash
git clone git@github.com:llvm/llvm-project
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
./x test --stage 1 tests/codegen-llvm/gpu_offload
```
