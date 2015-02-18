# The Rust Programming Language

This is a compiler for Rust, including standard libraries, tools and
documentation. Rust is a systems programming language that is fast,
memory safe and multithreaded, but does not employ a garbage collector
or otherwise impose significant runtime overhead.

## Quick Start

Read ["Installing Rust"] from [The Book].

["Installing Rust"]: http://doc.rust-lang.org/book/installing-rust.html
[The Book]: http://doc.rust-lang.org/book/index.html

## Building from Source

1. Make sure you have installed the dependencies:
    * `g++` 4.7 or `clang++` 3.x
    * `python` 2.6 or later (but not 3.x)
    * GNU `make` 3.81 or later
    * `curl`
    * `git`

2. Clone the [source] with `git`:

        $ git clone https://github.com/rust-lang/rust.git
        $ cd rust

[source]: https://github.com/rust-lang/rust

3. Build and install:

        $ ./configure
        $ make && make install

    > ***Note:*** You may need to use `sudo make install` if you do not normally have
    > permission to modify the destination directory. The install locations can
    > be adjusted by passing a `--prefix` argument to `configure`. Various other
    > options are also supported, pass `--help` for more information on them.

    When complete, `make install` will place several programs into
    `/usr/local/bin`: `rustc`, the Rust compiler, and `rustdoc`, the
    API-documentation tool. This install does not include [Cargo],
    Rust's package manager, which you may also want to build.

[Cargo]: https://github.com/rust-lang/cargo

### Building on Windows

To easily build on windows we can use [MSYS2](http://msys2.github.io/):

1. Grab the latest MSYS2 installer and go through the installer.
2. Now from the MSYS2 terminal we want to install the mingw64 toolchain and the other
   tools we need.

```bash
# choose one based on platform
$ pacman -S mingw-w64-i686-toolchain
$ pacman -S mingw-w64-x86_64-toolchain

$ pacman -S base-devel
```

3. With that now start `mingw32_shell.bat` or `mingw64_shell.bat`
   from where you installed MSYS2 (i.e. `C:\msys`). Which one you
   choose depends on if you want 32 or 64 bit Rust.
4. From there just navigate to where you have Rust's source code, configure and build it:

        $ ./configure
        $ make && make install

## Notes

Since the Rust compiler is written in Rust, it must be built by a
precompiled "snapshot" version of itself (made in an earlier state of
development). As such, source builds require a connection to the Internet, to
fetch snapshots, and an OS that can execute the available snapshot binaries.

Snapshot binaries are currently built and tested on several platforms:

* Windows (7, 8, Server 2008 R2), x86 and x86-64 (64-bit support added in Rust 0.12.0)
* Linux (2.6.18 or later, various distributions), x86 and x86-64
* OSX 10.7 (Lion) or greater, x86 and x86-64

You may find that other platforms work, but these are our officially
supported build environments that are most likely to work.

Rust currently needs about 1.5 GiB of RAM to build without swapping; if it hits
swap, it will take a very long time to build.

There is more advice about hacking on Rust in [CONTRIBUTING.md].

[CONTRIBUTING.md]: https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md

## Getting help

The Rust community congregates in a few places:

* [StackOverflow] - Direct questions about using the language here.
* [users.rust-lang.org] - General discussion, broader questions.
* [/r/rust] - News and general discussion.

[StackOverflow]: http://stackoverflow.com/questions/tagged/rust
[/r/rust]: http://reddit.com/r/rust
[users.rust-lang.org]: http://users.rust-lang.org/

## Contributing

To contribute to Rust, please see [CONTRIBUTING.md](CONTRIBUTING.md).

Rust has an [IRC] culture and most real-time collaboration happens in a
variety of channels on Mozilla's IRC network, irc.mozilla.org. The
most popular channel is [#rust], a venue for general discussion about
Rust, and a good place to ask for help,

[IRC]: https://en.wikipedia.org/wiki/Internet_Relay_Chat
[#rust]: irc://irc.mozilla.org/rust

## License

Rust is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See LICENSE-APACHE, LICENSE-MIT, and COPYRIGHT for details.
