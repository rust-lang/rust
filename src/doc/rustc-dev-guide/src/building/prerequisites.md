# Prerequisites

## Dependencies

Before building the compiler, you need the following things installed:

* `python` 3 or 2.7 (under the name `python`; `python2` or `python3` will not work)
* `curl`
* `git`
* `ssl` which comes in `libssl-dev` or `openssl-devel`
* `pkg-config` if you are compiling on Linux and targeting Linux

If building LLVM from source (the default), you'll need additional tools:

* `g++` 5.1 or later, `clang++` 3.5 or later, or MSVC 2017 or later.
* `ninja`, or GNU `make` 3.81 or later (ninja is recommended, especially on Windows)
* `cmake` 3.13.4 or later

Otherwise, you'll need LLVM installed and `llvm-config` in your path.
See [this section for more info][sysllvm].

[sysllvm]: ./suggested.md#skipping-llvm-build

### Windows

* Install [winget](https://github.com/microsoft/winget-cli)

`winget` is a Windows package manager. It will make package installation easy
on Windows.

Run the following in a terminal:

```powershell
winget install python
winget install cmake
```

If any of those is installed already, winget will detect it.
Then edit your systems `PATH` variable and add: `C:\Program Files\CMake\bin`.

For more information about building on Windows,
see [the `rust-lang/rust` README](https://github.com/rust-lang/rust#building-on-windows).

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
