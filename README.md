# Miri [![Build Status](https://travis-ci.org/solson/miri.svg?branch=master)](https://travis-ci.org/solson/miri) [![Windows build status](https://ci.appveyor.com/api/projects/status/github/solson/miri?svg=true)](https://ci.appveyor.com/project/solson63299/miri)


An experimental interpreter for [Rust][rust]'s
[mid-level intermediate representation][mir] (MIR).  It can run binaries and
test suites of cargo projects and detect certain classes of undefined behavior,
for example:

* Out-of-bounds memory accesses and use-after-free
* Invalid use of uninitialized data
* Violation of intrinsic preconditions (an [`unreachable_unchecked`] being
  reached, calling [`copy_nonoverlapping`] with overlapping ranges, ...)
* Not sufficiently aligned memory accesses and references
* Violation of basic type invariants (a `bool` that is not 0 or 1, for example,
  or an invalid enum discriminant)
* WIP: Violations of the rules governing aliasing for reference types

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html

## Building Miri

We recommend that you install [rustup] to obtain Rust. Then all you have
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

[rustup]: https://www.rustup.rs

## Running Miri on tiny examples

```sh
cargo +nightly run -- -Zmiri-disable-validation tests/run-pass/vecs.rs # Or whatever test you like.
```

We have to disable validation because that can lead to errors when libstd is not
compiled the right way.

## Running Miri on your own project('s test suite)

Install Miri as a cargo subcommand:

```sh
cargo +nightly install --git https://github.com/solson/miri/ miri
```

Be aware that if you used `rustup override set` to fix a particular Rust version
for the miri directory, that will *not* apply to your own project directory!
You have to use a consistent Rust version for building miri and your project for
this to work, so remember to either always specify the nightly version manually,
overriding it in your project directory as well, or use `rustup default nightly`
(or `rustup default nightly-YYYY-MM-DD`) to globally make `nightly` the default
toolchain.

1. Run `cargo clean` to eliminate any cached dependencies.  Miri needs your
   dependencies to be compiled the right way, that would not happen if they have
   previously already been compiled.
2. To run all tests in your project through Miri, use `cargo +nightly miri test`.
   **NOTE**: This is currently broken, see the discussion in
   [#479](https://github.com/solson/miri/issues/479).
3. If you have a binary project, you can run it through Miri using `cargo
   +nightly miri run`.

### Common Problems

When using the above instructions, you may encounter a number of confusing compiler
errors.

#### "found possibly newer version of crate `std` which `<dependency>` depends on"

Your build directory may contain artifacts from an earlier build that have/have
not been built for Miri. Run `cargo clean` before switching from non-Miri to
Miri builds and vice-versa.

#### "found crate `std` compiled by an incompatible version of rustc"

You may be running `cargo miri` with a different compiler version than the one
used to build the custom libstd that Miri uses, and Miri failed to detect that.
Try deleting `~/.cache/miri`.

## Miri `-Z` flags

Several `-Z` flags are relevant for miri:

* `-Zmir-opt-level` controls how many MIR optimizations are performed.  miri
  overrides the default to be `0`; be advised that using any higher level can
  make miri miss bugs in your program because they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic
  functions.  This is needed so that miri can execute such functions, so miri
  sets this flag per default.
* `-Zmiri-disable-validation` is a custom `-Z` flag added by miri.  It disables
  enforcing the validity invariant, which is enforced by default.  This is
  mostly useful for debugging; it means miri will miss bugs in your program.

## Development and Debugging

Since the heart of Miri (the main interpreter engine) lives in rustc, working on
Miri will often require using a locally built rustc. This includes getting a
trace of the execution, as distributed rustc has `debug!` and `trace!` disabled.

The first-time setup for a local rustc looks as follows:
```sh
git clone https://github.com/rust-lang/rust/ rustc
cd rustc
cp config.toml.example config.toml
# Now edit `config.toml` and set `debug-assertions = true` and `test-miri = true`.
# The latter is important to build libstd with the right flags for miri.
./x.py build src/rustc
# You may have to change the architecture in the next command
rustup toolchain link custom build/x86_64-unknown-linux-gnu/stage2
# Now cd to your Miri directory
rustup override set custom
```
The `build` step can take 30 minutes and more.

Now you can `cargo build` Miri, and you can `cargo test --release` it.  `cargo
test --release FILTER` only runs those tests that contain `FILTER` in their
filename (including the base directory, e.g. `cargo test --release fail` will
run all compile-fail tests).  We recommend using `--release` to make test
running take less time.

Notice that the "fullmir" tests only run if you have `MIRI_SYSROOT` set, the
test runner does not realized that your libstd comes with full MIR.  The
following will set it correctly:
```sh
MIRI_SYSROOT=$(rustc --print sysroot) cargo test --release
```

Moreover, you can now run Miri with a trace of all execution steps:
```sh
MIRI_LOG=debug cargo run tests/run-pass/vecs.rs
```

Setting `MIRI_LOG` like this will configure logging for miri itself as well as
the `rustc::mir::interpret` and `rustc_mir::interpret` modules in rustc.  You
can also do more targeted configuration, e.g. to debug the stacked borrows
implementation:
```sh
MIRI_LOG=rustc_mir::interpret=debug,miri::stacked_borrows cargo run tests/run-pass/vecs.rs
```

In addition, you can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally created.

If you changed something in rustc and want to re-build, run
```
./x.py --keep-stage 0 build src/rustc
```
This avoids rebuilding the entire stage 0, which can save a lot of time.

## Contributing and getting help

Check out the issues on this GitHub repository for some ideas. There's lots that
needs to be done that I haven't documented in the issues yet, however. For more
ideas or help with running or hacking on Miri, you can contact me (`scott`) on
Mozilla IRC in any of the Rust IRC channels (`#rust`, `#rust-offtopic`, etc).

## History

This project began as part of an undergraduate research course in 2015 by
@solson at the [University of Saskatchewan][usask].  There are [slides] and a
[report] available from that project.  In 2016, @oli-obk joined to prepare miri
for eventually being used as const evaluator in the Rust compiler itself
(basically, for `const` and `static` stuff), replacing the old evaluator that
worked directly on the AST.  In 2017, @RalfJung did an internship with Mozilla
and began developing miri towards a tool for detecting undefined behavior, and
also using miri as a way to explore the consequences of various possible
definitions for undefined behavior in Rust.  @oli-obk's move of the miri engine
into the compiler finally came to completion in early 2018.  Meanwhile, later
that year, @RalfJung did a second internship, developing miri further with
support for checking basic type invariants and verifying that references are
used according to their aliasing restrictions.

[usask]: https://www.usask.ca/
[slides]: https://solson.me/miri-slides.pdf
[report]: https://solson.me/miri-report.pdf

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
