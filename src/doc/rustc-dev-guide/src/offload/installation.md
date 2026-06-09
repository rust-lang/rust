# Installation

`std::offload` is partly available in nightly builds for users.
For now, everyone however still needs to build rustc from source to use all features of it.

## Build instructions

First you need to clone and configure the Rust repository:
```console
git clone git@github.com:rust-lang/rust
cd rust
./configure --enable-llvm-link-shared --release-channel=nightly --enable-llvm-assertions --enable-llvm-offload --enable-llvm-enzyme --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs
```

Afterwards you can build rustc using:
```console
./x build --stage 1 library
```

Afterwards rustc toolchain link will allow you to use it through cargo:
```console
rustup toolchain link offload build/host/stage1
rustup toolchain install nightly # enables -Z unstable-options
```



## Build instruction for LLVM itself
```console
git clone git@github.com:llvm/llvm-project
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_TARGETS_TO_BUILD="host;AMDGPU;NVPTX" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="offload;openmp" -DLLVM_ENABLE_PLUGINS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.
ninja
ninja install
```
This gives you a working LLVM build.


## Testing
Run this test script for offload-specific tests:
```console
./x test --stage 1 tests/codegen-llvm/gpu_offload
```

For testing the CI locally, you may use the commands outlined in [Testing with Docker](https://rustc-dev-guide.rust-lang.org/tests/docker.html):
```console
cargo run --manifest-path src/ci/citool/Cargo.toml run-local dist-x86_64-linux
```
This stores all compiler artifacts in the `obj` directory, however should you modify rustc-specific code, you may need to delete this directory as the Docker image will cache its state otherwise.

Submodules should also be checked out at this point.
