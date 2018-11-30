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


## Running Miri on your own project('s test suite)

Install Miri as a cargo subcommand:

```sh
cargo +nightly install --git https://github.com/solson/miri/ miri
```

If this does not work, try using the nightly version given in
[this file](https://raw.githubusercontent.com/solson/miri/master/rust-version). CI
should ensure that this nightly always works.

You have to use a consistent Rust version for building miri and your project, so
remember to either always specify the nightly version manually (like in the
example above), overriding it in your project directory as well, or use `rustup
default nightly` (or `rustup default nightly-YYYY-MM-DD`) to globally make
`nightly` the default toolchain.

Now you can run your project in miri:

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

## Development and Debugging

If you want to hack on miri yourself, great!  Here are some resources you might
find useful.

### Using a nightly rustc

miri heavily relies on internal rustc interfaces to execute MIR.  Still, some
things (like adding support for a new intrinsic) can be done by working just on
the miri side.

To prepare, make sure you are using a nightly Rust compiler.  You also need to
set up a libstd that enables execution with miri:

```sh
rustup override set nightly # or the nightly in `rust-version`
cargo run --bin cargo-miri -- miri setup
```

The last command should end in printing the directory where the libstd was
built.  Set that as your MIRI_SYSROOT environment variable:

```sh
export MIRI_SYSROOT=~/.cache/miri/HOST # or whatever the previous command said
```

### Testing Miri

Now you can run Miri directly, without going through `cargo miri`:

```sh
cargo run tests/run-pass-fullmir/format.rs # or whatever test you like
```

You can also run the test suite with `cargo test --release`.  `cargo test
--release FILTER` only runs those tests that contain `FILTER` in their filename
(including the base directory, e.g. `cargo test --release fail` will run all
compile-fail tests).  We recommend using `--release` to make test running take
less time.

Now you are set up!  You can write a failing test case, and tweak miri until it
fails no more.

### Using a locally built rustc

Since the heart of Miri (the main interpreter engine) lives in rustc, working on
Miri will often require using a locally built rustc.  The bug you want to fix
may actually be on the rustc side, or you just need to get more detailed trace
of the execution -- in both cases, you should develop miri against a rustc you
compiled yourself, with debug assertions (and hence tracing) enabled.

The setup for a local rustc works as follows:
```sh
git clone https://github.com/rust-lang/rust/ rustc
cd rustc
cp config.toml.example config.toml
# Now edit `config.toml` and set `debug-assertions = true` and `test-miri = true`.
# The latter is important to build libstd with the right flags for miri.
# This step can take 30 minutes and more.
./x.py build src/rustc
# If you change something, you can get a faster rebuild by doing
./x.py --keep-stage 0 build src/rustc
# You may have to change the architecture in the next command
rustup toolchain link custom build/x86_64-unknown-linux-gnu/stage2
# Now cd to your Miri directory, then configure rustup
rustup override set custom
# We also need to tell Miri where to find its sysroot. Since we set
# `test-miri` above, we can just use rustc' sysroot.
export MIRI_SYSROOT=$(rustc --print sysroot)
```

With this, you should now have a working development setup!  See "Testing Miri"
above for how to proceed.

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

### Miri `-Z` flags

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
