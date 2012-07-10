# The Rust Programming Language

This is a compiler for Rust, including standard libraries, tools and
documentation.


## Installation

The Rust compiler is slightly unusual in that it is written in Rust and
therefore must be built by a precompiled "snapshot" version of itself (made in
an earlier state of development). As such, source builds require that:

    * You are connected to the internet, to fetch snapshots.

    * You can at least execute snapshot binaries of one of the forms we offer
      them in. Currently we build and test snapshots on:
        * Windows (7, server 2008 r2) x86 only
        * Linux 2.6.x (various distributions) x86 and x86-64
        * OSX 10.6 ("Snow leopard") or 10.7 ("Lion") x86 and x86-64

You may find other platforms work, but these are our "tier 1" supported build
environments that are most likely to work. Further platforms will be added to
the list in the future via cross-compilation.

To build from source you will also need the following prerequisite packages:

    * g++ 4.4 or clang++ 3.x
    * python 2.6 or later
    * perl 5.0 or later
    * gnu make 3.81 or later
    * curl

Assuming you're on a relatively modern Linux/OSX system and have met the
prerequisites, something along these lines should work:

    $ tar -xzf rust-0.2.tar.gz
    $ cd rust-0.2
    $ ./configure
    $ make && make install

When complete, make install will place the following programs into
/usr/local/bin:

    * rustc, the Rust compiler
    * rustdoc, the API-documentation tool
    * cargo, the Rust package manager

In addition to a manual page under /usr/local/share/man and a set of host and
target libraries under /usr/local/lib/rustc.

The install locations can be adjusted by passing a --prefix argument to
configure. Various other options are also supported, pass --help for more
information on them.


## License

Rust is primarily distributed under the terms of the MIT license, with
portions covered by various BSD-like licenses.

See LICENSE.txt for complete terms of copyright and redistribution.


## More help

The [tutorial](http://dl.rust-lang.org/doc/tutorial.html) is a good
starting point.
