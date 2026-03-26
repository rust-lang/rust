# Installation

In the near future, `std::autodiff` should become available for users via rustup. As a rustc/enzyme/autodiff contributor however, you will still need to build rustc from source. 
For the meantime, you can download up-to-date builds to enable `std::autodiff` on your latest nightly toolchain, if you are using either of:  
**Linux**, with `x86_64-unknown-linux-gnu` or `aarch64-unknown-linux-gnu`  
**Windows**, with `x86_64-llvm-mingw` or `aarch64-llvm-mingw`  

You can also download slightly outdated builds for **Apple** (aarch64-apple), which should generally work for now. 

If you need any other platform, you can build rustc including autodiff from source. Please open an issue if you want to help enabling automatic builds for your prefered target.

## Installation guide

If you want to use `std::autodiff` and don't plan to contribute PR's to the project, then we recommend to just use your existing nightly installation and download the missing component. In the future, rustup will be able to do it for you.
For now, you'll have to manually download and copy it.

1) On our github repository, find the last merged PR: [`Repo`]
2) Scroll down to the lower end of the PR, where you'll find a rust-bors message saying `Test successful` with a `CI` link. 
3) Click on the `CI` link, and grep for your target. E.g. `dist-x86_64-linux` or `dist-aarch64-llvm-mingw` and click `Load summary`. 
4) Under the `CI artifacts` section, find the `enzyme-nightly` artifact, download, and unpack it.
5) Copy the artifact (libEnzyme-22.so for linux, libEnzyme-22.dylib for apple, etc.), which should be in a folder named `enzyme-preview`, to your rust toolchain directory. E.g. for linux: `cp  ~/Downloads/enzyme-nightly-x86_64-unknown-linux-gnu/enzyme-preview/lib/rustlib/x86_64-unknown-linux-gnu/lib/libEnzyme-22.so ~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/lib`

Apple support was temporarily reverted, due to downstream breakages. If you want to download autodiff for apple, please look at the artifacts from this [`run`].

## Installation guide for Nix user.

This setup was recommended by a nix and autodiff user. It uses [`Overlay`]. Please verify for yourself if you are comfortable using that repository.
In that case you might use the following nix configuration to get a rustc that supports `std::autodiff`.
```nix
{
  enzymeLib = pkgs.fetchzip {
    url = "https://ci-artifacts.rust-lang.org/rustc-builds/ec818fda361ca216eb186f5cf45131bd9c776bb4/enzyme-nightly-x86_64-unknown-linux-gnu.tar.xz";
    sha256 = "sha256-Rnrop44vzS+qmYNaRoMNNMFyAc3YsMnwdNGYMXpZ5VY=";
  };
  
  rustToolchain = pkgs.symlinkJoin {
    name = "rust-with-enzyme";
    paths = [pkgs.rust-bin.nightly.latest.default];
    nativeBuildInputs = [pkgs.makeWrapper];
    postBuild = ''
      libdir=$out/lib/rustlib/x86_64-unknown-linux-gnu/lib
      cp ${enzymeLib}/enzyme-preview/lib/rustlib/x86_64-unknown-linux-gnu/lib/libEnzyme-22.so $libdir/
      wrapProgram $out/bin/rustc --add-flags "--sysroot $out"
    '';
  };
}
```

## Build instructions

First you need to clone and configure the Rust repository. Based on your preferences, you might also want to `--enable-clang` or `--enable-lld`.
```bash
git clone git@github.com:rust-lang/rust
cd rust
./configure --release-channel=nightly --enable-llvm-enzyme --enable-llvm-link-shared --enable-llvm-assertions --enable-ninja --enable-option-checking --disable-docs --set llvm.download-ci-llvm=false
```

Afterwards you can build rustc using:
```bash
./x build --stage 1 library
```

Afterwards rustc toolchain link will allow you to use it through cargo:
```
rustup toolchain link enzyme build/host/stage1
rustup toolchain install nightly # enables -Z unstable-options
```

You can then run our test cases:

```bash
./x test --stage 1 tests/codegen-llvm/autodiff
./x test --stage 1 tests/pretty/autodiff
./x test --stage 1 tests/ui/autodiff
./x test --stage 1 tests/run-make/autodiff
./x test --stage 1 tests/ui/feature-gates/feature-gate-autodiff.rs
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
git clone https://github.com/rust-lang/rust
cd rust
./configure --release-channel=nightly --enable-llvm-enzyme --enable-llvm-link-shared --enable-llvm-assertions --enable-ninja --enable-option-checking --disable-docs --set llvm.download-ci-llvm=false
./x dist
```
We then copy the tarball to our host. The dockerid is the newest entry under `docker ps -a`.
```bash
docker cp <dockerid>:/rust/build/dist/rust-nightly-x86_64-unknown-linux-gnu.tar.gz rust-nightly-x86_64-unknown-linux-gnu.tar.gz
```
Afterwards we can create a new (pre-release) tag on the EnzymeAD/rust repository and make a PR against the EnzymeAD/enzyme-explorer repository to update the tag.
Remember to ping `tgymnich` on the PR to run his update script. Note: We should archive EnzymeAD/rust and update the instructions here. The explorer should soon
be able to get the rustc toolchain from the official rust servers.


## Build instruction for Enzyme itself

Following the Rust build instruction above will build LLVMEnzyme, LLDEnzyme, and ClangEnzyme along with the Rust compiler.
We recommend that approach, if you just want to use any of them and have no experience with cmake.
However, if you prefer to just build Enzyme without Rust, then these instructions might help.

```bash
git clone git@github.com:llvm/llvm-project
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
git clone git@github.com:EnzymeAD/Enzyme
cd Enzyme/enzyme
mkdir build 
cd build 
cmake .. -G Ninja -DLLVM_DIR=<YourLocalPath>/llvm-project/build/lib/cmake/llvm/ -DLLVM_EXTERNAL_LIT=<YourLocalPath>/llvm-project/llvm/utils/lit/lit.py -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -DBUILD_SHARED_LIBS=ON
ninja
```
This will build Enzyme, and you can find it in `Enzyme/enzyme/build/lib/<LLD/Clang/LLVM/lib>Enzyme.so`. (Endings might differ based on your OS).

[`Repo`]: https://github.com/rust-lang/rust/
[`run`]: https://github.com/rust-lang/rust/pull/153026#issuecomment-3950046599
[`Overlay`]: https://github.com/oxalica/rust-overlay
