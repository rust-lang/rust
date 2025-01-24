# Installing from Source

**Note: This document describes _building_ Rust _from source_.
This is _not recommended_ if you don't know what you're doing.
If you just want to install Rust, check out the [README.md](README.md) instead.**

The Rust build system uses a Python script called `x.py` to build the compiler,
which manages the bootstrapping process. It lives at the root of the project.
It also uses a file named `config.toml` to determine various configuration
settings for the build. You can see a full list of options in
`config.example.toml`.

The `x.py` command can be run directly on most Unix systems in the following
format:

```sh
./x.py <subcommand> [flags]
```

This is how the documentation and examples assume you are running `x.py`.
See the [rustc dev guide][rustcguidebuild] if this does not work on your
platform.

More information about `x.py` can be found by running it with the `--help` flag
or reading the [rustc dev guide][rustcguidebuild].

[gettingstarted]: https://rustc-dev-guide.rust-lang.org/getting-started.html
[rustcguidebuild]: https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html#what-is-xpy

## Dependencies

Make sure you have installed the dependencies:

* `python` 3 or 2.7
* `git`
* A C compiler (when building for the host, `cc` is enough; cross-compiling may
  need additional compilers)
* `curl` (not needed on Windows)
* `pkg-config` if you are compiling on Linux and targeting Linux
* `libiconv` (already included with glibc on Debian-based distros)

To build Cargo, you'll also need OpenSSL (`libssl-dev` or `openssl-devel` on
most Unix distros).

If building LLVM from source, you'll need additional tools:

* `g++`, `clang++`, or MSVC with versions listed on
  [LLVM's documentation](https://llvm.org/docs/GettingStarted.html#host-c-toolchain-both-compiler-and-standard-library)
* `ninja`, or GNU `make` 3.81 or later (Ninja is recommended, especially on
  Windows)
* `cmake` version listed on [LLVM's documentation](https://llvm.org/docs/GettingStarted.html#software)
* `libstdc++-static` may be required on some Linux distributions such as Fedora
  and Ubuntu

On tier 1 or tier 2 with host tools platforms, you can also choose to download
LLVM by setting `llvm.download-ci-llvm = true`.
Otherwise, you'll need LLVM installed and `llvm-config` in your path.
See [the rustc-dev-guide for more info][sysllvm].

[sysllvm]: https://rustc-dev-guide.rust-lang.org/building/new-target.html#using-pre-built-llvm


## Building on a Unix-like system

### Build steps

1. Clone the [source] with `git`:

   ```sh
   git clone https://github.com/rust-lang/rust.git
   cd rust
   ```

[source]: https://github.com/rust-lang/rust

2. Configure the build settings:

   ```sh
   ./configure
   ```

   If you plan to use `x.py install` to create an installation, you can either
   set `DESTDIR` environment variable to your custom directory path:

   ```bash
   export DESTDIR=<path>
   ```

   or set `prefix` and `sysconfdir` in the `[install]` section to your custom
   directory path:

   ```sh
   ./configure --set install.prefix=<path> --set install.sysconfdir=<path>
   ```

   When the `DESTDIR` environment variable is present, the `prefix` and
   `sysconfdir` values are combined with the path from the `DESTDIR`
   environment variable.

3. Build and install:

   ```sh
   ./x.py build && ./x.py install
   ```

   When complete, `./x.py install` will place several programs into
   `$PREFIX/bin`: `rustc`, the Rust compiler, and `rustdoc`, the
   API-documentation tool. By default, it will also include [Cargo], Rust's
   package manager. You can disable this behavior by passing
   `--set build.extended=false` to `./configure`.

[Cargo]: https://github.com/rust-lang/cargo

### Configure and Make

This project provides a configure script and makefile (the latter of which just
invokes `x.py`). `./configure` is the recommended way to programmatically
generate a `config.toml`. `make` is not recommended (we suggest using `x.py`
directly), but it is supported and we try not to break it unnecessarily.

```sh
./configure
make && sudo make install
```

`configure` generates a `config.toml` which can also be used with normal `x.py`
invocations.

## Building on Windows

On Windows, we suggest using [winget] to install dependencies by running the
following in a terminal:

```powershell
winget install -e Python.Python.3
winget install -e Kitware.CMake
winget install -e Git.Git
```

Then edit your system's `PATH` variable and add: `C:\Program Files\CMake\bin`.
See
[this guide on editing the system `PATH`](https://www.java.com/en/download/help/path.html)
from the Java documentation.

[winget]: https://github.com/microsoft/winget-cli

There are two prominent ABIs in use on Windows: the native (MSVC) ABI used by
Visual Studio and the GNU ABI used by the GCC toolchain. Which version of Rust
you need depends largely on what C/C++ libraries you want to interoperate with.
Use the MSVC build of Rust to interop with software produced by Visual Studio
and the GNU build to interop with GNU software built using the MinGW/MSYS2
toolchain.

### MinGW

[MSYS2][msys2] can be used to easily build Rust on Windows:

[msys2]: https://www.msys2.org/

1. Download the latest [MSYS2 installer][msys2] and go through the installer.

2. Download and install [Git for Windows](https://git-scm.com/download/win).
   Make sure that it's in your Windows PATH. To enable access to it from within
   MSYS2, edit the relevant `mingw[32|64].ini` file in your MSYS2 installation
   directory and uncomment the line `MSYS2_PATH_TYPE=inherit`.

   You could install and use MSYS2's version of git instead with `pacman`,
   however this is not recommended as it's excruciatingly slow, and not frequently
   tested for compatibility.

3. Start a MINGW64 or MINGW32 shell (depending on whether you want 32-bit
   or 64-bit Rust) either from your start menu, or by running `mingw64.exe`
   or `mingw32.exe` from your MSYS2 installation directory (e.g. `C:\msys64`).

4. From this terminal, install the required tools:

   ```sh
   # Update package mirrors (may be needed if you have a fresh install of MSYS2)
   pacman -Sy pacman-mirrors

   # Install build tools needed for Rust. If you're building a 32-bit compiler,
   # then replace "x86_64" below with "i686". 
   # Note that it is important that you do **not** use the 'python2', 'cmake',
   # and 'ninja' packages from the 'msys2' subsystem.
   # The build has historically been known to fail with these packages.
   pacman -S make \
               diffutils \
               tar \
               mingw-w64-x86_64-python \
               mingw-w64-x86_64-cmake \
               mingw-w64-x86_64-gcc \
               mingw-w64-x86_64-ninja
   ```

5. Navigate to Rust's source code (or clone it), then build it:

   ```sh
   python x.py setup dist && python x.py build && python x.py install
   ```

If you want to try the native Windows versions of Python or CMake, you can remove
them from the above pacman command and install them from another source. Follow
the instructions in step 2 to get them on PATH.

Using Windows native Python can be helpful if you get errors when building LLVM.
You may also want to use Git for Windows, as it is often *much* faster. Turning
off real-time protection in the Windows Virus & Threat protections settings can
also help with long run times (although note that it will automatically turn
itself back on after some time).

### MSVC

MSVC builds of Rust additionally require an installation of Visual Studio 2017
(or later) so `rustc` can use its linker. The simplest way is to get
[Visual Studio], check the "C++ build tools" and "Windows 10 SDK" workload.

[Visual Studio]: https://visualstudio.microsoft.com/downloads/

(If you're installing CMake yourself, be careful that "C++ CMake tools for
Windows" doesn't get included under "Individual components".)

With these dependencies installed, you can build the compiler in a `cmd.exe`
shell with:

```sh
python x.py setup user
python x.py build
```

Right now, building Rust only works with some known versions of Visual Studio.
If you have a more recent version installed and the build system doesn't
understand, you may need to force bootstrap to use an older version.
This can be done by manually calling the appropriate vcvars file before running
the bootstrap.

```batch
CALL "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
python x.py build
```

### Specifying an ABI

Each specific ABI can also be used from either environment (for example, using
the GNU ABI in PowerShell) by using an explicit build triple. The available
Windows build triples are:
- GNU ABI (using GCC)
    - `i686-pc-windows-gnu`
    - `x86_64-pc-windows-gnu`
- The MSVC ABI
    - `i686-pc-windows-msvc`
    - `x86_64-pc-windows-msvc`

The build triple can be specified by either specifying `--build=<triple>` when
invoking `x.py` commands, or by creating a `config.toml` file (as described in
[Building on a Unix-like system](#building-on-a-unix-like-system)), and passing
`--set build.build=<triple>` to `./configure`.

## Building Documentation

If you'd like to build the documentation, it's almost the same:

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
