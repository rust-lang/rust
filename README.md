# Miri [![Build Status](https://travis-ci.com/rust-lang/miri.svg?branch=master)](https://travis-ci.com/rust-lang/miri) [![Windows build status](https://ci.appveyor.com/api/projects/status/github/rust-lang/miri?svg=true)](https://ci.appveyor.com/project/rust-lang-libs/miri)


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

Miri has already discovered some [real-world bugs](#bugs-found-by-miri).  If you
found a bug with Miri, we'd appreciate if you tell us and we'll add it to the
list!

Be aware that Miri will not catch all possible errors in your program, and
cannot run all programs:

* There are still plenty of open questions around the basic invariants for some
  types and when these invariants even have to hold, so if you program runs fine
  in Miri right now that is by no means a guarantee that it is UB-free when
  these questions get answered.
* If the program relies on unspecified details of how data is laid out, it will
  still run fine in Miri -- but might break (including causing UB) on different
  compiler versions or different platforms.
* Miri is fully deterministic and does not actually pick a base address in
  virtual memory for the program's allocations.  If program behavior depends on
  the base address of an allocation, Miri will stop execution (with a few
  exceptions to make some common pointer comparisons work).
* Miri runs the program as a platform-independent interpreter, so the program
  has no access to any platform-specific APIs or FFI. A few APIs have been
  implemented (such as printing to stdout) but most have not: for example, Miri
  currently does not support concurrency, or networking, or file system access,
  or gathering entropy from the system.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html


## Running Miri on your own project (and its test suite)

Install Miri via `rustup`:

```sh
rustup component add miri
```

If `rustup` says the `miri` component is unavailable, that's because not all nightly releases come with all tools. Check out [this website](https://rust-lang.github.io/rustup-components-history) to determine a nightly version that comes with Miri and install that, e.g. using `rustup install nightly-2019-03-28`.

Now you can run your project in Miri:

1. Run `cargo clean` to eliminate any cached dependencies.  Miri needs your
   dependencies to be compiled the right way, that would not happen if they have
   previously already been compiled.
2. To run all tests in your project through Miri, use `cargo miri test`.
3. If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some
dependencies.  It will ask you for confirmation before installing anything.  If
you run Miri on CI, run `cargo miri setup` to avoid getting interactive
questions.

You can pass arguments to Miri after the first `--`, and pass arguments to the
interpreted program or test suite after the second `--`.  For example, `cargo
miri run -- -Zmiri-disable-validation` runs the program without validation of
basic type invariants and references.  `cargo miri test -- -- -Zunstable-options
--exclude-should-panic` skips `#[should_panic]` tests, which is a good idea
because Miri does not support unwinding or catching panics.

When running code via `cargo miri`, the `miri` config flag is set.  You can
use this to exclude test cases that will fail under Miri because they do things
Miri does not support:

```rust
#[cfg(not(miri))]
#[test]
fn does_not_work_on_miri() {
    let x = 0u8;
    assert!(&x as *const _ as usize % 4 < 4);
}
```

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

Miri heavily relies on internal rustc interfaces to execute MIR.  Still, some
things (like adding support for a new intrinsic) can be done by working just on
the Miri side.

To prepare, make sure you are using a nightly Rust compiler.  The most
convenient way is to install Miri using cargo, then you can easily run it on
other projects:

```sh
rustup component remove miri # avoid having Miri installed twice
cargo +nightly install --path "$DIR" --force
cargo +nightly miri setup
```

(We are giving `+nightly` explicitly here all the time because it is important
that all of these commands get executed with the same toolchain.)

In case this fails, your nightly might be incompatible with Miri master.  The
`rust-version` file contains the commit hash of rustc that Miri is currently
tested against; you can use that to find a nightly that works or you might have
to wait for the next nightly to get released.

If you want to use a different libstd (not the one that comes with the
nightly), you can do that by running

```sh
XARGO_RUST_SRC=~/src/rust/rustc/src/ cargo +nightly miri setup
```

Either way, you can now do `cargo +nightly miri run` to run Miri with your
local changes on whatever project you are debugging.

`cargo miri setup` should end in printing the directory where the libstd was
built.  For the next step to work, set that as your `MIRI_SYSROOT` environment
variable:

```sh
export MIRI_SYSROOT=~/.cache/miri/HOST # or whatever the previous command said
```

### Testing Miri

Instead of running an entire project using `cargo miri`, you can also use the
Miri "driver" directly to run just a single file.  That can be easier during
debugging.

```sh
cargo run tests/run-pass/format.rs # or whatever test you like
```

You can also run the test suite with `cargo test --release`.  `cargo test
--release FILTER` only runs those tests that contain `FILTER` in their filename
(including the base directory, e.g. `cargo test --release fail` will run all
compile-fail tests).  We recommend using `--release` to make test running take
less time.

Now you are set up!  You can write a failing test case, and tweak miri until it
fails no more.
You can get a trace of which MIR statements are being executed by setting the
`MIRI_LOG` environment variable.  For example:

```sh
MIRI_LOG=info cargo run tests/run-pass/vecs.rs
```

Setting `MIRI_LOG` like this will configure logging for miri itself as well as
the `rustc::mir::interpret` and `rustc_mir::interpret` modules in rustc.  You
can also do more targeted configuration, e.g. to debug the stacked borrows
implementation:
```sh
MIRI_LOG=rustc_mir::interpret=info,miri::stacked_borrows cargo run tests/run-pass/vecs.rs
```

In addition, you can set `MIRI_BACKTRACE=1` to get a backtrace of where an
evaluation error was originally created.


### Using a locally built rustc

Since the heart of Miri (the main interpreter engine) lives in rustc, working on
Miri will often require using a locally built rustc.  The bug you want to fix
may actually be on the rustc side, or you just need to get more detailed trace
of the execution than what is possible with release builds -- in both cases, you
should develop miri against a rustc you compiled yourself, with debug assertions
(and hence tracing) enabled.

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
```

With this, you should now have a working development setup!  See
["Testing Miri"](#testing-miri) above for how to proceed.

Running `cargo miri` in this setup is a bit more complicated, because the Miri
binary you just created does not actually run without some environment variables.
But you can contort cargo into calling `cargo miri` the right way for you:

```sh
# in some other project's directory, to run `cargo miri test`:
MIRI_SYSROOT=$(rustc +custom --print sysroot) cargo +custom run --manifest-path /path/to/miri/Cargo.toml --bin cargo-miri --release -- miri test
```

### Miri `-Z` flags and environment variables

Several `-Z` flags are relevant for Miri:

* `-Zmir-opt-level` controls how many MIR optimizations are performed.  miri
  overrides the default to be `0`; be advised that using any higher level can
  make miri miss bugs in your program because they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic
  functions.  This is needed so that miri can execute such functions, so miri
  sets this flag per default.
* `-Zmiri-disable-validation` is a custom `-Z` flag added by miri.  It disables
  enforcing the validity invariant, which is enforced by default.  This is
  mostly useful for debugging; it means miri will miss bugs in your program.

Moreover, Miri recognizes some environment variables:

* `MIRI_SYSROOT` (recognized by `miri`, `cargo miri` and the test suite)
  indicates the sysroot to use.
* `MIRI_TARGET` (recognized by the test suite) indicates which target
  architecture to test against.  `miri` and `cargo miri` accept the `--target`
  flag for the same purpose.

## Contributing and getting help

Check out the issues on this GitHub repository for some ideas. There's lots that
needs to be done that I haven't documented in the issues yet, however. For more
ideas or help with running or hacking on Miri, you can open an issue here on
GitHub or contact us (`oli-obk` and `RalfJ`) on the [Rust Zulip].

[Rust Zulip]: https://rust-lang.zulipchat.com

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

## Bugs found by Miri

Miri has already found a number of bugs in the Rust standard library, which we collect here.

* [`Debug for vec_deque::Iter` accessing uninitialized memory](https://github.com/rust-lang/rust/issues/53566)
* [`From<&[T]> for Rc` creating a not sufficiently aligned reference](https://github.com/rust-lang/rust/issues/54908)
* [`BTreeMap` creating a shared reference pointing to a too small allocation](https://github.com/rust-lang/rust/issues/54957)
* [`VecDeque` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/56161)
* [Futures turning a shared reference into a mutable one](https://github.com/rust-lang/rust/pull/56319)
* [`str` turning a shared reference into a mutable one](https://github.com/rust-lang/rust/pull/58200)
* [`BTreeMap` creating mutable references that overlap with shared references](https://github.com/rust-lang/rust/pull/58431)

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
