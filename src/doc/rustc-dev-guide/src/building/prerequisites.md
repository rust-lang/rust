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

[sysllvm]: ./new-target.md#using-pre-built-llvm

### Windows

* Install [winget](https://github.com/microsoft/winget-cli)

`winget` is a Windows package manager. It will make package installation easy
on Windows.

Run the following in a terminal:

```powershell
winget install -e Python.Python.3
winget install -e Kitware.CMake
```

If any of those is installed already, winget will detect it.
Then edit your systems `PATH` variable and add: `C:\Program Files\CMake\bin`.

For more information about building on Windows,
see [the `rust-lang/rust` README](https://github.com/rust-lang/rust#building-on-windows).

## Hardware

You will need an internet connection to build. The bootstrapping process
involves updating git submodules and downloading a beta compiler. It doesn't
need to be super fast, but that can help.

There are no strict hardware requirements, but building the compiler is
computationally expensive, so a beefier machine will help, and I wouldn't
recommend trying to build on a Raspberry Pi! We recommend the following.
* 30GB+ of free disk space. Otherwise, you will have to keep
  clearing incremental caches. More space is better, the compiler is a bit of a
  hog; it's a problem we are aware of.
* 8GB+ RAM
* 2+ cores. Having more cores really helps. 10 or 20 or more is not too many!

Beefier machines will lead to much faster builds. If your machine is not very
powerful, a common strategy is to only use `./x.py check` on your local machine
and let the CI build test your changes when you push to a PR branch.

Building the compiler takes more than half an hour on my moderately powerful
laptop. The first time you build the compiler, LLVM will also be built unless
you use CI-built LLVM ([see here][config]).

Like `cargo`, the build system will use as many cores as possible. Sometimes
this can cause you to run low on memory. You can use `-j` to adjust the number
concurrent jobs. If a full build takes more than ~45 minutes to an hour, you
are probably spending most of the time swapping memory in and out; try using
`-j1`.

If you don't have too much free disk space, you may want to turn off
incremental compilation ([see here][config]). This will make compilation take
longer (especially after a rebase), but will save a ton of space from the
incremental caches.

[config]: ./how-to-build-and-run.md#create-a-configtoml

## `rustc` and toolchain installation

Follow the installation given in the [Rust book][install] to install a working
`rustc` and the necessary C/++ toolchain on your platform.

[install]: https://doc.rust-lang.org/book/ch01-01-installation.html
