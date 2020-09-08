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
* **Experimental**: Violations of the [Stacked Borrows] rules governing aliasing
  for reference types

On top of that, Miri will also tell you about memory leaks: when there is memory
still allocated at the end of the execution, and that memory is not reachable
from a global `static`, Miri will raise an error. Note however that
[leak checking is currently disabled on Windows targets](https://github.com/rust-lang/miri/issues/1302).

Miri has already discovered some [real-world bugs](#bugs-found-by-miri).  If you
found a bug with Miri, we'd appreciate if you tell us and we'll add it to the
list!

However, be aware that Miri will **not catch all cases of undefined behavior**
in your program, and cannot run all programs:

* There are still plenty of open questions around the basic invariants for some
  types and when these invariants even have to hold. Miri tries to avoid false
  positives here, so if you program runs fine in Miri right now that is by no
  means a guarantee that it is UB-free when these questions get answered.

    In particular, Miri does currently not check that integers/floats are
  initialized or that references point to valid data.
* If the program relies on unspecified details of how data is laid out, it will
  still run fine in Miri -- but might break (including causing UB) on different
  compiler versions or different platforms.
* Program execution is non-deterministic when it depends, for example, on where
  exactly in memory allocations end up. Miri tests one of many possible
  executions of your program. If your code is sensitive to allocation base
  addresses or other non-deterministic data, try running Miri with different
  values for `-Zmiri-seed` to test different executions.
* Miri runs the program as a platform-independent interpreter, so the program
  has no access to most platform-specific APIs or FFI. A few APIs have been
  implemented (such as printing to stdout) but most have not: for example, Miri
  currently does not support SIMD or networking.
* Miri currently does not check for data-races and most other concurrency-related
  issues.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html
[Stacked Borrows]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md


## Using Miri

Install Miri on Rust nightly via `rustup`:

```sh
rustup +nightly component add miri
```

If `rustup` says the `miri` component is unavailable, that's because not all
nightly releases come with all tools. Check out
[this website](https://rust-lang.github.io/rustup-components-history) to
determine a nightly version that comes with Miri and install that using
`rustup toolchain install nightly-YYYY-MM-DD`.

Now you can run your project in Miri:

1. Run `cargo clean` to eliminate any cached dependencies.  Miri needs your
   dependencies to be compiled the right way, that would not happen if they have
   previously already been compiled.
2. To run all tests in your project through Miri, use `cargo miri test`.
3. If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some
dependencies.  It will ask you for confirmation before installing anything.

`cargo miri run/test` supports the exact same flags as `cargo run/test`.  You
can pass arguments to Miri via `MIRIFLAGS`. For example,
`MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo miri run` runs the program
without checking the aliasing of references.

Miri supports cross-execution: if you want to run the program as if it was a
Linux program, you can do `cargo miri run --target x86_64-unknown-linux-gnu`.
This is particularly useful if you are using Windows, as the Linux target is
much better supported than Windows targets.

When compiling code via `cargo miri`, the `cfg(miri)` config flag is set.  You
can use this to ignore test cases that fail under Miri because they do things
Miri does not support:

```rust
#[test]
#[cfg_attr(miri, ignore)]
fn does_not_work_on_miri() {
    std::thread::spawn(|| println!("Hello Thread!"))
        .join()
        .unwrap();
}
```

There is no way to list all the infinite things Miri cannot do, but the
interpreter will explicitly tell you when it finds something unsupported:

```
error: unsupported operation: can't call foreign function: bind
    ...
    = help: this is likely not a bug in the program; it indicates that the program \
            performed an operation that the interpreter does not support
```

### Running Miri on CI

To run Miri on CI, make sure that you handle the case where the latest nightly
does not ship the Miri component because it currently does not build.  For
example, you can use the following snippet to always test with the latest
nightly that *does* come with Miri:

```sh
MIRI_NIGHTLY=nightly-$(curl -s https://rust-lang.github.io/rustup-components-history/x86_64-unknown-linux-gnu/miri)
echo "Installing latest nightly with Miri: $MIRI_NIGHTLY"
rustup set profile minimal
rustup default "$MIRI_NIGHTLY"
rustup component add miri

cargo miri test
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

#### "no mir for `std::rt::lang_start_internal`"

This means the sysroot you are using was not compiled with Miri in mind.  This
should never happen when you use `cargo miri` because that takes care of setting
up the sysroot.  If you are using `miri` (the Miri driver) directly, see the
[contributors' guide](CONTRIBUTING.md) for how to use `./miri` to best do that.


## Miri `-Z` flags and environment variables
[miri-flags]: #miri--z-flags-and-environment-variables

Miri adds its own set of `-Z` flags, which are usually set via the `MIRIFLAGS`
environment variable:

* `-Zmiri-disable-alignment-check` disables checking pointer alignment, so you
  can focus on other failures, but it means Miri can miss bugs in your program.
  Using this flag is **unsound**.
* `-Zmiri-disable-stacked-borrows` disables checking the experimental
  [Stacked Borrows] aliasing rules.  This can make Miri run faster, but it also
  means no aliasing violations will be detected.  Using this flag is **unsound**
  (but the affected soundness rules are experimental).
* `-Zmiri-disable-validation` disables enforcing validity invariants, which are
  enforced by default.  This is mostly useful to focus on other failures (such
  as out-of-bounds accesses) first.  Setting this flag means Miri can miss bugs
  in your program.  However, this can also help to make Miri run faster.  Using
  this flag is **unsound**.
* `-Zmiri-disable-isolation` disables host isolation.  As a consequence,
  the program has access to host resources such as environment variables, file
  systems, and randomness.
* `-Zmiri-env-exclude=<var>` keeps the `var` environment variable isolated from
  the host so that it cannot be accessed by the program.  Can be used multiple
  times to exclude several variables.  On Windows, the `TERM` environment
  variable is excluded by default.
* `-Zmiri-ignore-leaks` disables the memory leak checker.
* `-Zmiri-seed=<hex>` configures the seed of the RNG that Miri uses to resolve
  non-determinism.  This RNG is used to pick base addresses for allocations.
  When isolation is enabled (the default), this is also used to emulate system
  entropy.  The default seed is 0.  **NOTE**: This entropy is not good enough
  for cryptographic use!  Do not generate secret keys in Miri or perform other
  kinds of cryptographic operations that rely on proper random numbers.
* `-Zmiri-symbolic-alignment-check` makes the alignment check more strict.  By
  default, alignment is checked by casting the pointer to an integer, and making
  sure that is a multiple of the alignment.  This can lead to cases where a
  program passes the alignment check by pure chance, because things "happened to
  be" sufficiently aligned -- there is no UB in this execution but there would
  be UB in others.  To avoid such cases, the symbolic alignment check only takes
  into account the requested alignment of the relevant allocation, and the
  offset into that allocation.  This avoids missing such bugs, but it also
  incurs some false positives when the code does manual integer arithmetic to
  ensure alignment.  (The standard library `align_to` method works fine in both
  modes; under symbolic alignment it only fills the middle slice when the
  allocation guarantees sufficient alignment.)
* `-Zmiri-track-alloc-id=<id>` shows a backtrace when the given allocation is
  being allocated or freed.  This helps in debugging memory leaks and
  use after free bugs.
* `-Zmiri-track-pointer-tag=<tag>` shows a backtrace when the given pointer tag
  is popped from a borrow stack (which is where the tag becomes invalid and any
  future use of it will error).  This helps you in finding out why UB is
  happening and where in your code would be a good place to look for it.
* `-Zmiri-track-call-id=<id>` shows a backtrace when the given call id is
  assigned to a stack frame.  This helps in debugging UB related to Stacked
  Borrows "protectors".

Some native rustc `-Z` flags are also very relevant for Miri:

* `-Zmir-opt-level` controls how many MIR optimizations are performed.  Miri
  overrides the default to be `0`; be advised that using any higher level can
  make Miri miss bugs in your program because they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic
  functions.  This is needed so that Miri can execute such functions, so Miri
  sets this flag per default.
* `-Zmir-emit-retag` controls whether `Retag` statements are emitted. Miri
  enables this per default because it is needed for [Stacked Borrows].

Moreover, Miri recognizes some environment variables:

* `MIRI_LOG`, `MIRI_BACKTRACE` control logging and backtrace printing during
  Miri executions, also [see above][testing-miri].
* `MIRIFLAGS` (recognized by `cargo miri` and the test suite) defines extra
  flags to be passed to Miri.
* `MIRI_SYSROOT` (recognized by `cargo miri` and the test suite)
  indicates the sysroot to use.  To do the same thing with `miri`
  directly, use the `--sysroot` flag.
* `MIRI_TEST_TARGET` (recognized by the test suite) indicates which target
  architecture to test against.  `miri` and `cargo miri` accept the `--target`
  flag for the same purpose.

The following environment variables are internal, but used to communicate between
different Miri binaries, and as such worth documenting:

* `MIRI_BE_RUSTC` when set to any value tells the Miri driver to actually not
  interpret the code but compile it like rustc would. This is useful to be sure
  that the compiled `rlib`s are compatible with Miri.

## Miri `extern` functions

Miri provides some `extern` functions that programs can import to access
Miri-specific functionality:

```rust
#[cfg(miri)]
extern "Rust" {
    /// Miri-provided extern function to mark the block `ptr` points to as a "root"
    /// for some static memory. This memory and everything reachable by it is not
    /// considered leaking even if it still exists when the program terminates.
    ///
    /// `ptr` has to point to the beginning of an allocated block.
    fn miri_static_root(ptr: *const u8);

    /// Miri-provided extern function to begin unwinding with the given payload.
    ///
    /// This is internal and unstable and should not be used; we give it here
    /// just to be complete.
    fn miri_start_panic(payload: *mut u8) -> !;
}
```

## Contributing and getting help

If you want to contribute to Miri, great!  Please check out our
[contribution guide](CONTRIBUTING.md).

For help with running Miri, you can open an issue here on
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
* [`Vec`](https://github.com/rust-lang/rust/issues/69770) and [`BTreeMap`](https://github.com/rust-lang/rust/issues/69769) leaking memory under some (panicky) conditions
* [`beef` leaking memory](https://github.com/maciejhirsz/beef/issues/12)
* [`EbrCell` using uninitialized memory incorrectly](https://github.com/Firstyear/concread/commit/b15be53b6ec076acb295a5c0483cdb4bf9be838f#diff-6282b2fc8e98bd089a1f0c86f648157cR229)
* [TiKV performing an unaligned pointer access](https://github.com/tikv/tikv/issues/7613)
* [`servo_arc` creating a dangling shared reference](https://github.com/servo/servo/issues/26357)
* [TiKV constructing out-of-bounds pointers (and overlapping mutable references)](https://github.com/tikv/tikv/pull/7751)
* [`encoding_rs` doing out-of-bounds pointer arithmetic](https://github.com/hsivonen/encoding_rs/pull/53)

Violations of [Stacked Borrows] found that are likely bugs (but Stacked Borrows is currently just an experiment):

* [`VecDeque` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/56161)
* [`BTreeMap` creating mutable references that overlap with shared references](https://github.com/rust-lang/rust/pull/58431)
* [`LinkedList` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/60072)
* [`Vec::push` invalidating existing references into the vector](https://github.com/rust-lang/rust/issues/60847)
* [`align_to_mut` violating uniqueness of mutable references](https://github.com/rust-lang/rust/issues/68549)
* [`sized-chunks` creating aliasing mutable references](https://github.com/bodil/sized-chunks/issues/8)
* [`String::push_str` invalidating existing references into the string](https://github.com/rust-lang/rust/issues/70301)
* [`ryu` using raw pointers outside their valid memory area](https://github.com/dtolnay/ryu/issues/24)
* [ink! creating overlapping mutable references](https://github.com/rust-lang/miri/issues/1364)
* [TiKV creating overlapping mutable reference and raw pointer](https://github.com/tikv/tikv/pull/7709)
* [Windows `Env` iterator creating `*const T` from `&T` to read memory outside of `T`](https://github.com/rust-lang/rust/pull/70479)

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
