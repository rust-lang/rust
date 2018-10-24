# Miri [[slides](https://solson.me/miri-slides.pdf)] [[report](https://solson.me/miri-report.pdf)] [![Build Status](https://travis-ci.org/solson/miri.svg?branch=master)](https://travis-ci.org/solson/miri) [![Windows build status](https://ci.appveyor.com/api/projects/status/github/solson/miri?svg=true)](https://ci.appveyor.com/project/solson63299/miri)


An experimental interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR). This project began as part of my work for the
undergraduate research course at the [University of Saskatchewan][usask].

## Building Miri

I recommend that you install [rustup][rustup] to obtain Rust. Then all you have
to do is:

```sh
cargo +nightly build
```

This uses the very latest Rust version.  If you experience any problem, refer to
the `rust-version` file which contains a particular Rust nightly version that
has been tested against the version of miri you are using.  Make sure to use
that particular `nightly-YYYY-MM-DD` whenever the instructions just say
`nightly`.

To avoid repeating the nightly version all the time, you can use
`rustup override set nightly` (or `rustup override set nightly-YYYY-MM-DD`),
which means `nightly` Rust will automatically be used whenever you are working
in this directory.

## Running Miri

```sh
cargo +nightly run tests/run-pass/vecs.rs # Or whatever test you like.
```

## Running Miri with full libstd

Per default libstd does not contain the MIR of non-polymorphic functions. When
Miri hits a call to such a function, execution terminates. To fix this, it is
possible to compile libstd with full MIR:

```sh
rustup component add --toolchain nightly rust-src
cargo +nightly install xargo
rustup run nightly xargo/build.sh
```

Now you can run Miri against the libstd compiled by xargo:

```sh
MIRI_SYSROOT=~/.xargo/HOST cargo +nightly run tests/run-pass-fullmir/hashmap.rs
```

Notice that you will have to re-run the last step of the preparations above
(`xargo/build.sh`) when your toolchain changes (e.g., when you update the
nightly).

## Running Miri on your own project('s test suite)

Install Miri as a cargo subcommand with `cargo install +nightly --all-features
--path .`.  Be aware that if you used `rustup override set` to fix a particular
Rust version for the miri directory, that will *not* apply to your own project
directory!  You have to use a consistent Rust version for building miri and your
project for this to work, so remember to either always specify the nightly
version manually, overriding it in your project directory as well, or use
`rustup default nightly` (or `rustup default nightly-YYYY-MM-DD`) to globally
make `nightly` the default toolchain.

We assume that you have prepared a MIR-enabled libstd as described above.  Now
compile your project and its dependencies against that libstd:

1. Run `cargo clean` to eliminate any cached dependencies that were built against
the non-MIR `libstd`.
2. To run all tests in your project through, Miri, use
`MIRI_SYSROOT=~/.xargo/HOST cargo +nightly miri test`.
3. If you have a binary project, you can run it through Miri using
`MIRI_SYSROOT=~/.xargo/HOST cargo +nightly miri`.

### Common Problems

When using the above instructions, you may encounter a number of confusing compiler
errors.

#### "constant evaluation error: no mir for `<function>`"

You may have forgotten to set `MIRI_SYSROOT` when calling `cargo miri`, and
your program called into `std` or `core`. Be sure to set `MIRI_SYSROOT=~/.xargo/HOST`.

#### "found possibly newer version of crate `std` which `<dependency>` depends on"

Your build directory may contain artifacts from an earlier build that did/did not
have `MIRI_SYSROOT` set. Run `cargo clean` before switching from non-Miri to Miri
builds and vice-versa.

#### "found crate `std` compiled by an incompatible version of rustc"

You may be running `cargo miri` with a different compiler version than the one
used to build the MIR-enabled `std`. Be sure to consistently use the same toolchain,
which should be the toolchain specified in the `rust-version` file.

## Miri `-Z` flags

Miri adds some extra `-Z` flags to control its behavior:

* `-Zmiri-start-fn`: This makes interpretation start with `lang_start` (defined
  in libstd) instead of starting with `main`.  Requires full MIR!
* `-Zmiri-disable-validation` disables enforcing the validity invariant.

## Development and Debugging

Since the heart of Miri (the main interpreter engine) lives in rustc, working on
Miri will often require using a locally built rustc. This includes getting a
trace of the execution, as distributed rustc has `trace!` disabled.

The first-time setup for a local rustc looks as follows:
```
git clone https://github.com/rust-lang/rust/ rustc
cd rustc
cp config.toml.example config.toml
# Now edit `config.toml` and set `debug-assertions = true`
./x.py build src/rustc
# You may have to change the architecture in the next command
rustup toolchain link custom build/x86_64-unknown-linux-gnu/stage2
# Now cd to your Miri directory
rustup override set custom
```
The `build` step can take 30 minutes and more.

Now you can `cargo build` Miri, and you can `cargo test --tests`.  (`--tests`
is needed to skip doctests because we have not built rustdoc for your custom
toolchain.) You can also set `RUST_LOG=rustc_mir::interpret=trace` as
environment variable to get a step-by-step trace.

If you changed something in rustc and want to re-build, run
```
./x.py build src/rustc --keep-stage 0
```
This avoids rebuilding the entire stage 0, which can save a lot of time.

## Contributing and getting help

Check out the issues on this GitHub repository for some ideas. There's lots that
needs to be done that I haven't documented in the issues yet, however. For more
ideas or help with running or hacking on Miri, you can contact me (`scott`) on
Mozilla IRC in any of the Rust IRC channels (`#rust`, `#rust-offtopic`, etc).

## License

Licensed under either of
  * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or
    http://opensource.org/licenses/MIT) at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[usask]: https://www.usask.ca/
[rustup]: https://www.rustup.rs
