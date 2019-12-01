# Contribution Guide

If you want to hack on miri yourself, great!  Here are some resources you might
find useful.

## Getting started

Check out the issues on this GitHub repository for some ideas. There's lots that
needs to be done that we haven't documented in the issues yet, however. For more
ideas or help with hacking on Miri, you can contact us (`oli-obk` and `RalfJ`)
on the [Rust Zulip].

[Rust Zulip]: https://rust-lang.zulipchat.com

## Building Miri with a pre-built rustc

Miri heavily relies on internal rustc interfaces to execute MIR.  Still, some
things (like adding support for a new intrinsic or a shim for an external
function being called) can be done by working just on the Miri side.

The `rust-version` file contains the commit hash of rustc that Miri is currently
tested against. Other versions will likely not work. After installing
[`rustup-toolchain-install-master`], you can run the following command to
install that exact version of rustc as a toolchain:
```
./rustup-toolchain
```

[`rustup-toolchain-install-master`]: https://github.com/kennytm/rustup-toolchain-install-master

### Fixing Miri when rustc changes

Miri is heavily tied to rustc internals, so it is very common that rustc changes
break Miri.  Fixing those is a good way to get starting working on Miri.
Usually, Miri will require changes similar to the other consumers of the changed
rustc API, so reading the rustc PR diff is a good way to get an idea for what is
needed.

To update the `rustc-version` file and install the latest rustc, you can run:
```
./rustup-toolchain HEAD
```

Now try `./miri test`, and submit a PR once that works again.

## Testing the Miri driver
[testing-miri]: #testing-the-miri-driver

The Miri driver in the `miri` binary is the "heart" of Miri: it is basically a
version of `rustc` that, instead of compiling your code, runs it.  It accepts
all the same flags as `rustc` (though the ones only affecting code generation
and linking obviously will have no effect) [and more][miri-flags].

Running the Miri driver requires some fiddling with environment variables, so
the `miri` script helps you do that.  For example, you can run the driver on a
particular file by doing

```sh
./miri run tests/run-pass/format.rs
./miri run tests/run-pass/hello.rs --target i686-unknown-linux-gnu
```

and you can run the test suite using:

```
./miri test
```

`./miri test FILTER` only runs those tests that contain `FILTER` in their
filename (including the base directory, e.g. `./miri test fail` will run all
compile-fail tests).

You can get a trace of which MIR statements are being executed by setting the
`MIRI_LOG` environment variable.  For example:

```sh
MIRI_LOG=info ./miri run tests/run-pass/vecs.rs
```

Setting `MIRI_LOG` like this will configure logging for Miri itself as well as
the `rustc::mir::interpret` and `rustc_mir::interpret` modules in rustc.  You
can also do more targeted configuration, e.g. the following helps debug the
stacked borrows implementation:

```sh
MIRI_LOG=rustc_mir::interpret=info,miri::stacked_borrows ./miri run tests/run-pass/vecs.rs
```

In addition, you can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally raised.

## Testing `cargo miri`

Working with the driver directly gives you full control, but you also lose all
the convenience provided by cargo.  Once your test case depends on a crate, it
is probably easier to test it with the cargo wrapper.  You can install your
development version of Miri using

```
./miri install
```

and then you can use it as if it was installed by `rustup`.  Make sure you use
the same toolchain when calling `cargo miri` that you used when installing Miri!

There's a test for the cargo wrapper in the `test-cargo-miri` directory; run
`./run-test.py` in there to execute it.

## Building Miri with a locally built rustc

A big part of the Miri driver lives in rustc, so working on Miri will sometimes
require using a locally built rustc.  The bug you want to fix may actually be on
the rustc side, or you just need to get more detailed trace of the execution
than what is possible with release builds -- in both cases, you should develop
miri against a rustc you compiled yourself, with debug assertions (and hence
tracing) enabled.

The setup for a local rustc works as follows:
```sh
git clone https://github.com/rust-lang/rust/ rustc
cd rustc
cp config.toml.example config.toml
# Now edit `config.toml` and set `debug-assertions = true`.
# This step can take 30 minutes and more.
./x.py build src/rustc
# If you change something, you can get a faster rebuild by doing
./x.py --keep-stage 0 build src/rustc
# You may have to change the architecture in the next command
rustup toolchain link custom build/x86_64-unknown-linux-gnu/stage2
# Now cd to your Miri directory, then configure rustup
rustup override set custom
```

With this, you should now have a working development setup!  See
[above][testing-miri] for how to proceed working with the Miri driver.
