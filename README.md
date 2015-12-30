# The Rust Programming Language

This is the main source code repository for [Rust]. It contains the compiler, standard library,
and documentation.

[Rust]: https://www.rust-lang.org

## Quick Start

Read ["Installing Rust"] from [The Book].

["Installing Rust"]: https://doc.rust-lang.org/book/installing-rust.html
[The Book]: https://doc.rust-lang.org/book/index.html

## Building from Source

1. Make sure you have installed the dependencies:

   * `g++` 4.7 or `clang++` 3.x
   * `python` 2.6 or later (but not 3.x)
   * GNU `make` 3.81 or later
   * `curl`
   * `git`

2. Clone the [source] with `git`:

   ```sh
   $ git clone --depth=1 https://github.com/rust-lang/rust.git
   $ cd rust
   ```

[source]: https://github.com/rust-lang/rust

3. Build and install:

    ```sh
    $ ./configure
    $ make && make install
    ```

    > ***Note:*** You may need to use `sudo make install` if you do not
    > normally have permission to modify the destination directory. The
    > install locations can be adjusted by passing a `--prefix` argument
    > to `configure`. Various other options are also supported – pass
    > `--help` for more information on them.

    When complete, `make install` will place several programs into
    `/usr/local/bin`: `rustc`, the Rust compiler, and `rustdoc`, the
    API-documentation tool. This install does not include [Cargo],
    Rust's package manager, which you may also want to build.

[Cargo]: https://github.com/rust-lang/cargo

### Building on Windows

[MSYS2](http://msys2.github.io/) can be used to easily build Rust on Windows:

1. Grab the latest MSYS2 installer and go through the installer.

2. From the MSYS2 terminal, install the `mingw64` toolchain and other required
   tools.

   ```sh
   # Update package mirrors (may be needed if you have a fresh install of MSYS2)
   $ pacman -Sy pacman-mirrors

   # Choose one based on platform: 
   # *** see the note below ***
   $ pacman -S mingw-w64-i686-toolchain
   $ pacman -S mingw-w64-x86_64-toolchain

   # Make git available in MSYS2 (if not already available on path)
   $ pacman -S git

   $ pacman -S base-devel
   ```

3. Run `mingw32_shell.bat` or `mingw64_shell.bat` from wherever you installed
   MSYS2 (i.e. `C:\msys`), depending on whether you want 32-bit or 64-bit Rust.

4. Navigate to Rust's source code, configure and build it:

   ```sh
   $ ./configure
   $ make && make install
   ```
> ***Note:*** gcc versions >= 5 currently have issues building LLVM on Windows
> resulting in a segmentation fault when building Rust. In order to avoid this
> it may be necessary to obtain an earlier version of gcc such as 4.9.x.  
> Msys's `pacman` will install the latest version, so for the time being it is
> recommended to skip gcc toolchain installation step above and use [Mingw-Builds]
> project's installer instead.  Be sure to add gcc `bin` directory to the path
> before running `configure`.  
> For more information on this see issue #28260.

[Mingw-Builds]: http://sourceforge.net/projects/mingw-w64/

## Building Documentation

If you’d like to build the documentation, it’s almost the same:

```sh
./configure
$ make docs
```

Building the documentation requires building the compiler, so the above
details will apply. Once you have the compiler built, you can

```sh
$ make docs NO_REBUILD=1
```

To make sure you don’t re-build the compiler because you made a change
to some documentation.

The generated documentation will appear in a top-level `doc` directory,
created by the `make` rule.

## Notes

Since the Rust compiler is written in Rust, it must be built by a
precompiled "snapshot" version of itself (made in an earlier state of
development). As such, source builds require a connection to the Internet, to
fetch snapshots, and an OS that can execute the available snapshot binaries.

Snapshot binaries are currently built and tested on several platforms:

| Platform \ Architecture        | x86 | x86_64 |
|--------------------------------|-----|--------|
| Windows (7, 8, Server 2008 R2) | ✓   | ✓      |
| Linux (2.6.18 or later)        | ✓   | ✓      |
| OSX (10.7 Lion or later)       | ✓   | ✓      |

You may find that other platforms work, but these are our officially
supported build environments that are most likely to work.

Rust currently needs between 600MiB and 1.5GiB to build, depending on platform. If it hits
swap, it will take a very long time to build.

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
Rust, and a good place to ask for help.

[IRC]: https://en.wikipedia.org/wiki/Internet_Relay_Chat
[#rust]: irc://irc.mozilla.org/rust

## License

Rust is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT), and [COPYRIGHT](COPYRIGHT) for details.
