# Prerequisites

Before building the compiler, you need the following things installed:

* Python
* A C/C++ compiler toolchain
* cmake
* rustc

## `rustc` and toolchain installation

Follow the installation given in the [Rust book](https://doc.rust-lang.org/book/ch01-01-installation.html) to install a working `rustc` and the necessary C/++ toolchain on your platform.

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
