# The Rust Programming Language

This is the main source code repository for [Rust]. It contains the compiler,
standard library, and documentation.

[Rust]: https://www.rust-lang.org

## Quick Start

Read ["Installing Rust"] from [The Book].

["Installing Rust"]: https://doc.rust-lang.org/book/getting-started.html#installing-rust
[The Book]: https://doc.rust-lang.org/book/index.html

## Building from Source

1. Make sure you have installed the dependencies:

   * `g++` 4.7 or later or `clang++` 3.x
   * `python` 2.7 (but not 3.x)
   * GNU `make` 3.81 or later
   * `cmake` 3.4.3 or later
   * `curl`
   * `git`

2. Clone the [source] with `git`:

   ```sh
   $ git clone https://github.com/rust-lang/rust.git
   $ cd rust
   ```

[source]: https://github.com/rust-lang/rust

3. Build and install:

    ```sh
    $ ./configure
    $ make && sudo make install
    ```

    > ***Note:*** Install locations can be adjusted by passing a `--prefix`
    > argument to `configure`. Various other options are also supported – pass
    > `--help` for more information on them.

    When complete, `sudo make install` will place several programs into
    `/usr/local/bin`: `rustc`, the Rust compiler, and `rustdoc`, the
    API-documentation tool. This install does not include [Cargo],
    Rust's package manager, which you may also want to build.

[Cargo]: https://github.com/rust-lang/cargo

### Building on Windows

There are two prominent ABIs in use on Windows: the native (MSVC) ABI used by
Visual Studio, and the GNU ABI used by the GCC toolchain. Which version of Rust
you need depends largely on what C/C++ libraries you want to interoperate with:
for interop with software produced by Visual Studio use the MSVC build of Rust;
for interop with GNU software built using the MinGW/MSYS2 toolchain use the GNU
build.


#### MinGW

[MSYS2][msys2] can be used to easily build Rust on Windows:

[msys2]: https://msys2.github.io/

1. Grab the latest [MSYS2 installer][msys2] and go through the installer.

2. Run `mingw32_shell.bat` or `mingw64_shell.bat` from wherever you installed
   MSYS2 (i.e. `C:\msys64`), depending on whether you want 32-bit or 64-bit
   Rust. (As of the latest version of MSYS2 you have to run `msys2_shell.cmd
   -mingw32` or `msys2_shell.cmd -mingw64` from the command line instead)

3. From this terminal, install the required tools:

   ```sh
   # Update package mirrors (may be needed if you have a fresh install of MSYS2)
   $ pacman -Sy pacman-mirrors

   # Install build tools needed for Rust. If you're building a 32-bit compiler,
   # then replace "x86_64" below with "i686". If you've already got git, python,
   # or CMake installed and in PATH you can remove them from this list. Note
   # that it is important that you do **not** use the 'python2' and 'cmake'
   # packages from the 'msys2' subsystem. The build has historically been known
   # to fail with these packages.
   $ pacman -S git \
               make \
               diffutils \
               tar \
               mingw-w64-x86_64-python2 \
               mingw-w64-x86_64-cmake \
               mingw-w64-x86_64-gcc
   ```

4. Navigate to Rust's source code (or clone it), then configure and build it:

   ```sh
   $ ./configure
   $ make && make install
   ```

#### MSVC

MSVC builds of Rust additionally require an installation of Visual Studio 2013
(or later) so `rustc` can use its linker. Make sure to check the “C++ tools”
option.

With these dependencies installed, you can build the compiler in a `cmd.exe`
shell with:

```sh
> python x.py build
```

If you're running inside of an msys shell, however, you can run:

```sh
$ ./configure --build=x86_64-pc-windows-msvc
$ make && make install
```

Currently building Rust only works with some known versions of Visual Studio. If
you have a more recent version installed the build system doesn't understand
then you may need to force rustbuild to use an older version. This can be done
by manually calling the appropriate vcvars file before running the bootstrap.

```
CALL "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin\amd64\vcvars64.bat"
python x.py build
```

## Building Documentation

If you’d like to build the documentation, it’s almost the same:

```sh
$ ./configure
$ make docs
```

The generated documentation will appear in a top-level `doc` directory,
created by the `make` rule.

## Notes

Since the Rust compiler is written in Rust, it must be built by a
precompiled "snapshot" version of itself (made in an earlier state of
development). As such, source builds require a connection to the Internet, to
fetch snapshots, and an OS that can execute the available snapshot binaries.

Snapshot binaries are currently built and tested on several platforms:

| Platform / Architecture        | x86 | x86_64 |
|--------------------------------|-----|--------|
| Windows (7, 8, Server 2008 R2) | ✓   | ✓      |
| Linux (2.6.18 or later)        | ✓   | ✓      |
| OSX (10.7 Lion or later)       | ✓   | ✓      |

You may find that other platforms work, but these are our officially
supported build environments that are most likely to work.

Rust currently needs between 600MiB and 1.5GiB to build, depending on platform.
If it hits swap, it will take a very long time to build.

There is more advice about hacking on Rust in [CONTRIBUTING.md].

[CONTRIBUTING.md]: https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md

## Getting Help

The Rust community congregates in a few places:

* [Stack Overflow] - Direct questions about using the language.
* [users.rust-lang.org] - General discussion and broader questions.
* [/r/rust] - News and general discussion.

[Stack Overflow]: http://stackoverflow.com/questions/tagged/rust
[/r/rust]: http://reddit.com/r/rust
[users.rust-lang.org]: https://users.rust-lang.org/

## Contributing

To contribute to Rust, please see [CONTRIBUTING](CONTRIBUTING.md).

Rust has an [IRC] culture and most real-time collaboration happens in a
variety of channels on Mozilla's IRC network, irc.mozilla.org. The
most popular channel is [#rust], a venue for general discussion about
Rust. And a good place to ask for help would be [#rust-beginners].

[IRC]: https://en.wikipedia.org/wiki/Internet_Relay_Chat
[#rust]: irc://irc.mozilla.org/rust
[#rust-beginners]: irc://irc.mozilla.org/rust-beginners

## License

Rust is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT), and
[COPYRIGHT](COPYRIGHT) for details.
