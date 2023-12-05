# Rust Programming Language

    [![Rust Community](https://img.shields.io/badge/Rust_Community%20-Join_us-brightgreen?style=plastic&logo=rust)](https://www.rust-lang.org/community)

    Welcome to the official repository for the Rust programming language. This repository contains the Rust compiler, standard library, and documentation.

    **Note**: This README is for users of Rust. If you want to contribute to the development of the Rust compiler, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file.

    ## Table of Contents
    - [Quick Start](#quick-start)
    - [Installing from Source](#installing-from-source)
    - [Building Documentation](#building-documentation)
    - [Notes](#notes)
    - [Getting Help](#getting-help)
    - [Contributing](#contributing)
    - [License](#license)
    - [Trademark](#trademark)

    ## Quick Start

    For a quick start guide to installing Rust, read the ["Installation" section](https://doc.rust-lang.org/book/ch01-01-installation.html) in [The Rust Programming Language book](https://doc.rust-lang.org/book/index.html).

    ## Installing from Source

    To build Rust from source, you can use the `x.py` Python script. It manages the bootstrapping process and uses a `config.toml` file to configure the build. You can find a full list of configuration options in `config.example.toml`.

    On Unix systems, you can run `x.py` with the following command:

    ```sh
    ./x.py <subcommand> [flags]
    ```

    For detailed information on using `x.py`, run `./x.py --help` or refer to the [rustc dev guide](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html#what-is-xpy).

    ### Dependencies

    Before building Rust, make sure you have the following dependencies installed:
    - `python` 3 or 2.7
    - `git`
    - A C compiler (for the host, `cc` is sufficient; cross-compiling may require additional compilers)
    - `curl` (not needed on Windows)
    - `pkg-config` (for Linux when targeting Linux)
    - `libiconv` (included with glibc on Debian-based distros)

    To build Cargo, you'll also need OpenSSL (`libssl-dev` or `openssl-devel` on most Unix distros).

    If building LLVM from source, additional tools are required, including `g++`, `clang++`, `ninja`, or GNU `make`, and `cmake`. For some Linux distributions, you may need `libstdc++-static`.

    You can download LLVM by setting `llvm.download-ci-llvm = true` on tier 1 or tier 2 platforms with host tools.

    For platform-specific instructions, see the [rustc dev guide](https://rustc-dev-guide.rust-lang.org/getting-started.html).

    ### Building on Unix-like Systems

    #### Build Steps

    1. Clone the Rust source repository with `git`:

    ```sh
    git clone https://github.com/rust-lang/rust.git
    cd rust
    ```

    2. Configure the build settings:

    ```sh
    ./configure
    ```

    3. Build and install:

    ```sh
    ./x.py build && ./x.py install
    ```

    When complete, `./x.py install` will place `rustc` and `rustdoc` in `$PREFIX/bin`. By default, it will also include [Cargo], Rust's package manager.

    #### Configure and Make

    You can use the configure script to generate a `config.toml`. For make-based builds, follow these steps:

    ```sh
    ./configure
    make && sudo make install
    ```

    ### Building on Windows

    On Windows, we recommend using [winget] to install dependencies:

    ```powershell
    winget install -e Python.Python.3
    winget install -e Kitware.CMake
    winget install -e Git.Git
    ```

    For more information on Windows-specific builds, please refer to the documentation.

    ### Specifying an ABI

    You can specify the Windows build ABI using the `--build` flag or by creating a `config.toml` file.

    Available Windows build triples:
    - GNU ABI: `i686-pc-windows-gnu` and `x86_64-pc-windows-gnu`
    - MSVC ABI: `i686-pc-windows-msvc` and `x86_64-pc-windows-msvc`

    ## Building Documentation

    To build Rust documentation, use the following command:

    ```sh
    ./x.py doc
    ```


The generated documentation will appear under `doc` in the `build` directory for
the ABI used. That is, if the ABI was `x86_64-pc-windows-msvc`, the directory
will be `build\x86_64-pc-windows-msvc\doc`.

## Notes

Since the Rust compiler is written in Rust, it must be built by a precompiled
"snapshot" version of itself (made in an earlier stage of development).
As such, source builds require an Internet connection to fetch snapshots, and an
OS that can execute the available snapshot binaries.

See https://doc.rust-lang.org/nightly/rustc/platform-support.html for a list of
supported platforms.
Only "host tools" platforms have a pre-compiled snapshot binary available; to
compile for a platform without host tools you must cross-compile.

You may find that other platforms work, but these are our officially supported
build environments that are most likely to work.

## Getting Help

See https://www.rust-lang.org/community for a list of chat platforms and forums.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Rust is primarily distributed under the terms of both the MIT license and the
Apache License (Version 2.0), with portions covered by various BSD-like
licenses.

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.

## Trademark

[The Rust Foundation][rust-foundation] owns and protects the Rust and Cargo
trademarks and logos (the "Rust Trademarks").

If you want to use these names or brands, please read the
[media guide][media-guide].

Third-party logos may be subject to third-party copyrights and trademarks. See
[Licenses][policies-licenses] for details.

[rust-foundation]: https://foundation.rust-lang.org/
[media-guide]: https://foundation.rust-lang.org/policies/logo-policy-and-media-guide/
[policies-licenses]: https://www.rust-lang.org/policies/licenses
