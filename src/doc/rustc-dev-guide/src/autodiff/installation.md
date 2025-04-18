# Installation

In the near future, `std::autodiff` should become available in nightly builds for users. As a contribute however, you will still need to build rustc from source. Please be aware that the msvc target is not supported at the moment, all other tier 1 targets should work. Please open an issue if you encounter any problems on a supported tier 1 target, or if you succesfully build this project on a tier2/tier3 target.

## Build instructions

First you need to clone and configure the Rust repository:
```bash
git clone --depth=1 git@github.com:rust-lang/rust.git
cd rust
./configure --enable-llvm-link-shared --enable-llvm-plugins --enable-llvm-enzyme --release-channel=nightly --enable-llvm-assertions --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs
```

Afterwards you can build rustc using:
```bash
./x.py build --stage 1 library
```

Afterwards rustc toolchain link will allow you to use it through cargo:
```
rustup toolchain link enzyme build/host/stage1
rustup toolchain install nightly # enables -Z unstable-options
```

You can then run our test cases:

```bash
./x.py test --stage 1 library tests/ui/autodiff
./x.py test --stage 1 library tests/codegen/autodiff
./x.py test --stage 1 library tests/pretty/autodiff*
```

Autodiff is still experimental, so if you want to use it in your own projects, you will need to add `lto="fat"` to your Cargo.toml 
and use `RUSTFLAGS="-Zautodiff=Enable" cargo +enzyme` instead of `cargo` or `cargo +nightly`. 

## Compiler Explorer and dist builds

Our compiler explorer instance can be updated to a newer rustc in a similar way. First, prepare a docker instance.
```bash
docker run -it ubuntu:22.04
export CC=clang CXX=clang++
apt update
apt install wget vim python3 git curl libssl-dev pkg-config lld ninja-build cmake clang build-essential 
```
Then build rustc in a slightly altered way:
```bash
git clone --depth=1 https://github.com/EnzymeAD/rust.git
cd rust
./configure --enable-llvm-link-shared --enable-llvm-plugins --enable-llvm-enzyme --release-channel=nightly --enable-llvm-assertions --enable-clang --enable-lld --enable-option-checking --enable-ninja --disable-docs
./x dist
```
We then copy the tarball to our host. The dockerid is the newest entry under `docker ps -a`.
```bash
docker cp <dockerid>:/rust/build/dist/rust-nightly-x86_64-unknown-linux-gnu.tar.gz rust-nightly-x86_64-unknown-linux-gnu.tar.gz
```
Afterwards we can create a new (pre-release) tag on the EnzymeAD/rust repository and make a PR against the EnzymeAD/enzyme-explorer repository to update the tag.
Remember to ping `tgymnich` on the PR to run his update script.


## Build instruction for Enzyme itself

Following the Rust build instruction above will build LLVMEnzyme, LLDEnzyme, and ClangEnzyme along with the Rust compiler.
We recommend that approach, if you just want to use any of them and have no experience with cmake.
However, if you prefer to just build Enzyme without Rust, then these instructions might help.

```bash
git clone --depth=1 git@github.com:llvm/llvm-project.git 
cd llvm-project
mkdir build
cd build
cmake -G Ninja ../llvm -DLLVM_TARGETS_TO_BUILD="host" -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_PROJECTS="clang;lld" -DLLVM_ENABLE_RUNTIMES="openmp" -DLLVM_ENABLE_PLUGINS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=.
ninja
ninja install
```
This gives you a working LLVM build, now we can continue with building Enzyme.
Leave the `llvm-project` folder, and execute the following commands:
```bash
git clone git@github.com:EnzymeAD/Enzyme.git 
cd Enzyme/enzyme
mkdir build 
cd build 
cmake .. -G Ninja -DLLVM_DIR=<YourLocalPath>/llvm-project/build/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<YourLocalPath>/llvm-project/llvm/utils/lit/lit.py -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DBUILD_SHARED_LIBS=ON
ninja
```
This will build Enzyme, and you can find it in `Enzyme/enzyme/build/lib/<LLD/Clang/LLVM>Enzyme.so`. (Endings might differ based on your OS).

