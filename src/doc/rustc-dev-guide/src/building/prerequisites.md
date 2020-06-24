# Prerequisites

## Dependencies

Before building the compiler, you need the following things installed:

* `g++` 5.1 or later or `clang++` 3.5 or later
* `python` 3 or 2.7
* GNU `make` 3.81 or later
* `cmake` 3.4.3 or later
* `curl`
* `git`
* `ssl` which comes in `libssl-dev` or `openssl-devel`
* `pkg-config` if you are compiling on Linux and targeting Linux

Additionally, if you want to build `rustc` with your system's LLVM, you will
need `llvm-config`. See [this section for more info][sysllvm].

[sysllvm]: ./suggested.md#building-with-system-llvm

## Hardware

These are not so much requirements as _recommendations_:

* ~15GB of free disk space (~25GB or more if doing incremental builds).
* \>= 8GB RAM
* \>= 4 cores
* Internet access

Beefier machines will lead to much faster builds. If your machine is not very
powerful, a common strategy is to only use `./x.py check` on your local machine
and let the CI build test your changes when you push to a PR branch.

## `rustc` and toolchain installation

Follow the installation given in the [Rust book][install] to install a working
`rustc` and the necessary C/++ toolchain on your platform.

[install]: https://doc.rust-lang.org/book/ch01-01-installation.html

## Platform specific instructions

### Windows

* Install [winget](https://github.com/microsoft/winget-cli)

Run the following in a terminal:

```
winget install python
winget install cmake
```

If any of those is installed already, winget will detect it.

Edit your systems `PATH` variable and add: `C:\Program Files\CMake\bin`.
