# The Rust Programming Language

This is a compiler for Rust, including standard libraries, tools and
documentation.


## Installation

The Rust compiler currently must be built from a [tarball], unless you
are on Windows, in which case using the [installer][win-exe] is
recommended.

Since the Rust compiler is written in Rust, it must be built by
a precompiled "snapshot" version of itself (made in an earlier state
of development). As such, source builds require a connection to
the Internet, to fetch snapshots, and an OS that can execute the
available snapshot binaries.

Snapshot binaries are currently built and tested on several platforms:

* Windows (7, Server 2008 R2), x86 only
* Linux (various distributions), x86 and x86-64
* OSX 10.6 ("Snow Leopard") or greater, x86 and x86-64

You may find that other platforms work, but these are our "tier 1"
supported build environments that are most likely to work.

> ***Note:*** Windows users should read the detailed
> [getting started][wiki-start] notes on the wiki. Even when using
> the binary installer the Windows build requires a MinGW installation,
> the precise details of which are not discussed here.

To build from source you will also need the following prerequisite
packages:

* g++ 4.4 or clang++ 3.x
* python 2.6 or later (but not 3.x)
* perl 5.0 or later
* gnu make 3.81 or later
* curl

Assuming you're on a relatively modern *nix system and have met the
prerequisites, something along these lines should work.

    $ curl -O http://static.rust-lang.org/dist/rust-0.6.tar.gz
    $ tar -xzf rust-0.6.tar.gz
    $ cd rust-0.6
    $ ./configure
    $ make && make install

You may need to use `sudo make install` if you do not normally have
permission to modify the destination directory. The install locations
can be adjusted by passing a `--prefix` argument to
`configure`. Various other options are also supported, pass `--help`
for more information on them.

When complete, `make install` will place several programs into
`/usr/local/bin`: `rustc`, the Rust compiler; `rustdoc`, the
API-documentation tool, and `rustpkg`, the Rust package manager and build system.

[wiki-start]: https://github.com/mozilla/rust/wiki/Note-getting-started-developing-Rust
[tarball]: http://static.rust-lang.org/dist/rust-0.6.tar.gz
[win-exe]: http://static.rust-lang.org/dist/rust-0.6-install.exe


## License

Rust is primarily distributed under the terms of both the MIT license
and the Apache License (Version 2.0), with portions covered by various
BSD-like licenses.

See LICENSE-APACHE, LICENSE-MIT, and COPYRIGHT for details.

## More help

The [tutorial] is a good starting point.

[tutorial]: http://static.rust-lang.org/doc/tutorial.html
