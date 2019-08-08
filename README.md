# Miri [![Build Status](https://travis-ci.com/rust-lang/miri.svg?branch=master)](https://travis-ci.com/rust-lang/miri) [![Windows build status](https://ci.appveyor.com/api/projects/status/github/rust-lang/miri?svg=true)](https://ci.appveyor.com/project/rust-lang-libs/miri)


An experimental interpreter for [Rust][rust]'s
[mid-level intermediate representation][mir] (MIR).  It can run binaries and
test suites of cargo projects and detect certain classes of
[undefined behavior](https://doc.rust-lang.org/reference/behavior-considered-undefined.html),
for example:

* Out-of-bounds memory accesses and use-after-free
* Invalid use of uninitialized data
* Violation of intrinsic preconditions (an [`unreachable_unchecked`] being
  reached, calling [`copy_nonoverlapping`] with overlapping ranges, ...)
* Not sufficiently aligned memory accesses and references
* Violation of *some* basic type invariants (a `bool` that is not 0 or 1, for example,
  or an invalid enum discriminant)
* WIP: Violations of the rules governing aliasing for reference types

Miri has already discovered some [real-world bugs](#bugs-found-by-miri).  If you
found a bug with Miri, we'd appreciate if you tell us and we'll add it to the
list!

Be aware that Miri will not catch all cases of undefined behavior in your
program, and cannot run all programs:

* There are still plenty of open questions around the basic invariants for some
  types and when these invariants even have to hold. Miri tries to avoid false
  positives here, so if you program runs fine in Miri right now that is by no
  means a guarantee that it is UB-free when these questions get answered.

    In particular, Miri does currently not check that integers are initialized
  or that references point to valid data.
* If the program relies on unspecified details of how data is laid out, it will
  still run fine in Miri -- but might break (including causing UB) on different
  compiler versions or different platforms.
* Program execution is non-deterministic when it depends, for example, on where
  exactly in memory allocations end up. Miri tests one of many possible
  executions of your program. If your code is sensitive to allocation base
  addresses or other non-deterministic data, try running Miri with different
  values for `-Zmiri-seed` to test different executions.
* Miri runs the program as a platform-independent interpreter, so the program
  has no access to any platform-specific APIs or FFI. A few APIs have been
  implemented (such as printing to stdout) but most have not: for example, Miri
  currently does not support concurrency, or SIMD, or networking, or file system
  access.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html


## Running Miri on your own project (and its test suite)

Install Miri via `rustup`:

```sh
rustup component add miri
```

If `rustup` says the `miri` component is unavailable, that's because not all
nightly releases come with all tools. Check out
[this website](https://rust-lang.github.io/rustup-components-history) to
determine a nightly version that comes with Miri and install that, e.g. using
`rustup install nightly-2019-03-28`.

Now you can run your project in Miri:

1. Run `cargo clean` to eliminate any cached dependencies.  Miri needs your
   dependencies to be compiled the right way, that would not happen if they have
   previously already been compiled.
2. To run all tests in your project through Miri, use `cargo miri test`.
3. If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some
dependencies.  It will ask you for confirmation before installing anything.

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

### Running Miri on CI

To run Miri on CI, make sure that you handle the case where the latest nightly
does not ship the Miri component because it currently does not build.  For
example, you can use the following snippet to always test with the latest
nightly that *does* come with Miri:

```sh
MIRI_NIGHTLY=nightly-$(curl -s https://rust-lang.github.io/rustup-components-history/x86_64-unknown-linux-gnu/miri)
echo "Installing latest nightly with Miri: $MIRI_NIGHTLY"
rustup default "$MIRI_NIGHTLY"

rustup component add miri
cargo miri setup

cargo miri test -- -- -Zunstable-options --exclude-should-panic
```

We use `cargo miri setup` to avoid getting interactive questions about the extra
setup needed for Miri.

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

#### "no mir for `std::rt::lang_start_internal`"

This means the sysroot you are using was not compiled with Miri in mind.  This
should never happen when you use `cargo miri` because that takes care of setting
up the sysroot.  If you are using `miri` (the Miri driver) directly, see
[below][testing-miri] for how to set up the sysroot.


## Miri `-Z` flags and environment variables
[miri-flags]: #miri--z-flags-and-environment-variables

Several `-Z` flags are relevant for Miri:

* `-Zmiri-seed=<hex>` is a custom `-Z` flag added by Miri.  It configures the
  seed of the RNG that Miri uses to resolve non-determinism.  This RNG is used
  to pick base addresses for allocations, and when the interpreted program
  requests system entropy.  The default seed is 0.
  **NOTE**: This entropy is not good enough for cryptographic use!  Do not
  generate secret keys in Miri or perform other kinds of cryptographic
  operations that rely on proper random numbers.
* `-Zmiri-disable-validation` disables enforcing the validity invariant, which
  is enforced by default.  This is mostly useful for debugging; it means Miri
  will miss bugs in your program.  However, this can also help to make Miri run
  faster.
* `-Zmir-opt-level` controls how many MIR optimizations are performed.  Miri
  overrides the default to be `0`; be advised that using any higher level can
  make Miri miss bugs in your program because they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic
  functions.  This is needed so that Miri can execute such functions, so Miri
  sets this flag per default.
* `-Zmir-emit-retag` controls whether `Retag` statements are emitted. Miri
  enables this per default because it is needed for validation.

Moreover, Miri recognizes some environment variables:

* `MIRI_LOG`, `MIRI_BACKTRACE` control logging and backtrace printing during
  Miri executions, also [see above][testing-miri].
* `MIRI_SYSROOT` (recognized by `cargo miri` and the test suite)
  indicates the sysroot to use.  To do the same thing with `miri`
  directly, use the `--sysroot` flag.
* `MIRI_TEST_TARGET` (recognized by the test suite) indicates which target
  architecture to test against.  `miri` and `cargo miri` accept the `--target`
  flag for the same purpose.

## Development and Debugging

If you want to hack on miri yourself, great!  Here are some resources you might
find useful.

### Using a nightly rustc

Miri heavily relies on internal rustc interfaces to execute MIR.  Still, some
things (like adding support for a new intrinsic or a shim for an external
function being called) can be done by working just on the Miri side.

To prepare, make sure you are using a nightly Rust compiler.  Then you should be
able to just `cargo build` Miri.

In case this fails, your nightly might be incompatible with Miri master.  The
`rust-version` file contains the commit hash of rustc that Miri is currently
tested against; you can use that to find a nightly that works or you might have
to wait for the next nightly to get released. You can also use
[`rustup-toolchain-install-master`](https://github.com/kennytm/rustup-toolchain-install-master)
to install that exact version of rustc as a toolchain:
```
rustup-toolchain-install-master $(cat rust-version) -c rust-src
```

Another common problem is outdated dependencies: Miri does not come with a
lockfile (it cannot, due to how it gets embedded into the rustc build). So you
have to run `cargo update` every now and then yourself to make sure you are
using the latest versions of everything (which is what gets tested on CI).

### Testing the Miri driver
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

### Testing `cargo miri`

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

### Using a locally built rustc

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

Miri has already found a number of bugs in the Rust standard library and beyond, which we collect here.

Definite bugs found:

* [`Debug for vec_deque::Iter` accessing uninitialized memory](https://github.com/rust-lang/rust/issues/53566)
* [`Vec::into_iter` doing an unaligned ZST read](https://github.com/rust-lang/rust/pull/53804)
* [`From<&[T]> for Rc` creating a not sufficiently aligned reference](https://github.com/rust-lang/rust/issues/54908)
* [`BTreeMap` creating a shared reference pointing to a too small allocation](https://github.com/rust-lang/rust/issues/54957)
* [`Vec::append` creating a dangling reference](https://github.com/rust-lang/rust/pull/61082)
* [Futures turning a shared reference into a mutable one](https://github.com/rust-lang/rust/pull/56319)
* [`str` turning a shared reference into a mutable one](https://github.com/rust-lang/rust/pull/58200)
* [`rand` performing unaligned reads](https://github.com/rust-random/rand/issues/779)
* [The Unix allocator calling `posix_memalign` in an invalid way](https://github.com/rust-lang/rust/issues/62251)
* [`getrandom` calling the `getrandom` syscall in an invalid way](https://github.com/rust-random/getrandom/pull/73)

Violations of Stacked Borrows found that are likely bugs (but Stacked Borrows is currently just an experiment):

* [`VecDeque` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/56161)
* [`BTreeMap` creating mutable references that overlap with shared references](https://github.com/rust-lang/rust/pull/58431)
* [`LinkedList` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/60072)
* [`Vec::push` invalidating existing references into the vector](https://github.com/rust-lang/rust/issues/60847)

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
