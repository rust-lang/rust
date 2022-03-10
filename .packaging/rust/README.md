# Enzyme build helper

## Goal

This repository will build enzyme/llvm/clang/rustc in the right configuration such that you can use it in combination with [oxide-enzyme](https://github.com/rust-ml/oxide-enzyme).

## Requirements

 - git  
 - ninja  
 - cmake  
 - libssl-dev
 - libclang-dev
 - Rust (rustup) with an installed nightly toolchain   
 - ~10GB free storage in $HOME/.cache

## Usage

Build LLVM, the Rust toolchain and Enzyme with

```sh
cargo install enzyme && enzyme-install --rust-stable --enzyme-stable
```

Depending on your CPU this might take a few hours.  
The build process will run enzyme tests, so your last output should look similar to these lines:

Testing Time: 0.63s  
  Passed           : 576  
  Expectedly Failed:   5  

## Release schedule:
We will automatically release a new version of this crate every time either a new Rust, or Enzyme version is published.
If you want to experiment with upcomming or local changes you can also use --enzyme-head or --enzyme-local \<Path\> (same goes for the Rust flag).

## Extras
- Q: It fails some (all) tests or the build breaks even earlier. Help?
- A: Sorry. Please create a github issue with relevant information (OS, error message) [here](https://github.com/EnzymeAD/Enzyme/issues) or open a topic on our [Discourse](https://discourse.llvm.org/c/projects-that-want-to-become-official-llvm-projects/enzyme/45).
As an alternative you can also ping us on [Discord](https://discord.gg/MGBqckV7Zb).  
&nbsp;
- Q: How often do I have to run this? It takes quite a while..
- A: We are aware of this and working on offering pre-build versions.

License
=======

Dual-licensed to be compatible with the Rust project.

Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0 or the MIT license
http://opensource.org/licenses/MIT, at your
option. This file may not be copied, modified, or distributed
except according to those terms.
