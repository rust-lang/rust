# Miri

An experimental interpreter for [Rust][rust]'s
[mid-level intermediate representation][mir] (MIR). It can run binaries and
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
* **Experimental**: Violations of the [Tree Borrows] aliasing rules, as an optional
  alternative to [Stacked Borrows]
* **Experimental**: Data races

On top of that, Miri will also tell you about memory leaks: when there is memory
still allocated at the end of the execution, and that memory is not reachable
from a global `static`, Miri will raise an error.

Miri supports almost all Rust language features; in particular, unwinding and
concurrency are properly supported (including some experimental emulation of
weak memory effects, i.e., reads can return outdated values).

You can use Miri to emulate programs on other targets, e.g. to ensure that
byte-level data manipulation works correctly both on little-endian and
big-endian systems. See
[cross-interpretation](#cross-interpretation-running-for-different-targets)
below.

Miri has already discovered some [real-world bugs](#bugs-found-by-miri). If you
found a bug with Miri, we'd appreciate if you tell us and we'll add it to the
list!

By default, Miri ensures a fully deterministic execution and isolates the
program from the host system. Some APIs that would usually access the host, such
as gathering entropy for random number generators, environment variables, and
clocks, are replaced by deterministic "fake" implementations. Set
`MIRIFLAGS="-Zmiri-disable-isolation"` to access the real system APIs instead.
(In particular, the "fake" system RNG APIs make Miri **not suited for
cryptographic use**! Do not generate keys using Miri.)

All that said, be aware that Miri will **not catch all cases of undefined
behavior** in your program, and cannot run all programs:

* There are still plenty of open questions around the basic invariants for some
  types and when these invariants even have to hold. Miri tries to avoid false
  positives here, so if your program runs fine in Miri right now that is by no
  means a guarantee that it is UB-free when these questions get answered.

    In particular, Miri does currently not check that references point to valid data.
* If the program relies on unspecified details of how data is laid out, it will
  still run fine in Miri -- but might break (including causing UB) on different
  compiler versions or different platforms.
* Program execution is non-deterministic when it depends, for example, on where
  exactly in memory allocations end up, or on the exact interleaving of
  concurrent threads. Miri tests one of many possible executions of your
  program. You can alleviate this to some extent by running Miri with different
  values for `-Zmiri-seed`, but that will still by far not explore all possible
  executions.
* Miri runs the program as a platform-independent interpreter, so the program
  has no access to most platform-specific APIs or FFI. A few APIs have been
  implemented (such as printing to stdout, accessing environment variables, and
  basic file system access) but most have not: for example, Miri currently does
  not support networking. System API support varies between targets; if you run
  on Windows it is a good idea to use `--target x86_64-unknown-linux-gnu` to get
  better support.
* Weak memory emulation may [produce weak behaviours](https://github.com/rust-lang/miri/issues/2301)
  unobservable by compiled programs running on real hardware when `SeqCst` fences are used, and it
  cannot produce all behaviors possibly observable on real hardware.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html
[Stacked Borrows]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[Tree Borrows]: https://perso.crans.org/vanille/treebor/


## Using Miri

Install Miri on Rust nightly via `rustup`:

```sh
rustup +nightly component add miri
```

If `rustup` says the `miri` component is unavailable, that's because not all
nightly releases come with all tools. Check out
[this website](https://rust-lang.github.io/rustup-components-history) to
determine a nightly version that comes with Miri and install that using `rustup
toolchain install nightly-YYYY-MM-DD`. Either way, all of the following commands
assume the right toolchain is pinned via `rustup override set nightly` or
`rustup override set nightly-YYYY-MM-DD`. (Alternatively, use `cargo
+nightly`/`cargo +nightly-YYYY-MM-DD` for each of the following commands.)

Now you can run your project in Miri:

1. Run `cargo clean` to eliminate any cached dependencies. Miri needs your
   dependencies to be compiled the right way, that would not happen if they have
   previously already been compiled.
2. To run all tests in your project through Miri, use `cargo miri test`.
3. If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some
dependencies. It will ask you for confirmation before installing anything.

`cargo miri run/test` supports the exact same flags as `cargo run/test`. For
example, `cargo miri test filter` only runs the tests containing `filter` in
their name.

You can pass arguments to Miri via `MIRIFLAGS`. For example,
`MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo miri run` runs the program
without checking the aliasing of references.

When compiling code via `cargo miri`, the `cfg(miri)` config flag is set for code
that will be interpret under Miri. You can use this to ignore test cases that fail
under Miri because they do things Miri does not support:

```rust
#[test]
#[cfg_attr(miri, ignore)]
fn does_not_work_on_miri() {
    tokio::run(futures::future::ok::<_, ()>(()));
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

### Cross-interpretation: running for different targets

Miri can not only run a binary or test suite for your host target, it can also
perform cross-interpretation for arbitrary foreign targets: `cargo miri run
--target x86_64-unknown-linux-gnu` will run your program as if it was a Linux
program, no matter your host OS. This is particularly useful if you are using
Windows, as the Linux target is much better supported than Windows targets.

You can also use this to test platforms with different properties than your host
platform. For example `cargo miri test --target mips64-unknown-linux-gnuabi64`
will run your test suite on a big-endian target, which is useful for testing
endian-sensitive code.

### Running Miri on CI

To run Miri on CI, make sure that you handle the case where the latest nightly
does not ship the Miri component because it currently does not build. `rustup
toolchain install --component` knows how to handle this situation, so the
following snippet should always work:

```sh
rustup toolchain install nightly --component miri
rustup override set nightly

cargo miri test
```

Here is an example job for GitHub Actions:

```yaml
  miri:
    name: "Miri"
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install Miri
        run: |
          rustup toolchain install nightly --component miri
          rustup override set nightly
          cargo miri setup
      - name: Test with Miri
        run: cargo miri test
```

The explicit `cargo miri setup` helps to keep the output of the actual test step
clean.

### Testing for alignment issues

Miri can sometimes miss misaligned accesses since allocations can "happen to be"
aligned just right. You can use `-Zmiri-symbolic-alignment-check` to definitely
catch all such issues, but that flag will also cause false positives when code
does manual pointer arithmetic to account for alignment. Another alternative is
to call Miri with various values for `-Zmiri-seed`; that will alter the
randomness that is used to determine allocation base addresses. The following
snippet calls Miri in a loop with different values for the seed:

```
for SEED in $(seq 0 255); do
  echo "Trying seed: $SEED"
  MIRIFLAGS=-Zmiri-seed=$SEED cargo miri test || { echo "Failing seed: $SEED"; break; };
done
```

### Supported targets

Miri does not support all targets supported by Rust. The good news, however, is
that no matter your host OS/platform, it is easy to run code for *any* target
using `--target`!

The following targets are tested on CI and thus should always work (to the
degree documented below):

- The best-supported target is `x86_64-unknown-linux-gnu`. Miri releases are
  blocked on things working with this target. Most other Linux targets should
  also work well; we do run the test suite on `i686-unknown-linux-gnu` as a
  32bit target and `mips64-unknown-linux-gnuabi64` as a big-endian target, as
  well as the ARM targets `aarch64-unknown-linux-gnu` and
  `arm-unknown-linux-gnueabi`.
- `x86_64-apple-darwin` should work basically as well as Linux. We also test
  `aarch64-apple-darwin`. However, we might ship Miri with a nightly even when
  some features on these targets regress.
- `x86_64-pc-windows-msvc` works, but supports fewer features than the Linux and
  Apple targets. For example, file system access and concurrency are not
  supported on Windows. We also test `i686-pc-windows-msvc`, with the same
  reduced feature set. We might ship Miri with a nightly even when some features
  on these targets regress.

### Running tests in parallel

Though it implements Rust threading, Miri itself is a single-threaded interpreter.
This means that when running `cargo miri test`, you will probably see a dramatic
increase in the amount of time it takes to run your whole test suite due to the
inherent interpreter slowdown and a loss of parallelism.

You can get your test suite's parallelism back by running `cargo miri nextest run -jN`
(note that you will need [`cargo-nextest`](https://nexte.st) installed).
This works because `cargo-nextest` collects a list of all tests then launches a
separate `cargo miri run` for each test. You will need to specify a `-j` or `--test-threads`;
by default `cargo miri nextest run` runs one test at a time. For more details, see the
[`cargo-nextest` Miri documentation](https://nexte.st/book/miri.html).

Note: This one-test-per-process model means that `cargo miri test` is able to detect data
races where two tests race on a shared resource, but `cargo miri nextest run` will not detect
such races.

Note: `cargo-nextest` does not support doctests, see https://github.com/nextest-rs/nextest/issues/16

### Common Problems

When using the above instructions, you may encounter a number of confusing compiler
errors.

#### "note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace"

You may see this when trying to get Miri to display a backtrace. By default, Miri
doesn't expose any environment to the program, so running
`RUST_BACKTRACE=1 cargo miri test` will not do what you expect.

To get a backtrace, you need to disable isolation
[using `-Zmiri-disable-isolation`][miri-flags]:

```sh
RUST_BACKTRACE=1 MIRIFLAGS="-Zmiri-disable-isolation" cargo miri test
```

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
environment variable. We first document the most relevant and most commonly used flags:

* `-Zmiri-compare-exchange-weak-failure-rate=<rate>` changes the failure rate of
  `compare_exchange_weak` operations. The default is `0.8` (so 4 out of 5 weak ops will fail).
  You can change it to any value between `0.0` and `1.0`, where `1.0` means it
  will always fail and `0.0` means it will never fail. Note than setting it to
  `1.0` will likely cause hangs, since it means programs using
  `compare_exchange_weak` cannot make progress.
* `-Zmiri-disable-isolation` disables host isolation.  As a consequence,
  the program has access to host resources such as environment variables, file
  systems, and randomness.
* `-Zmiri-disable-leak-backtraces` disables backtraces reports for memory leaks. By default, a
  backtrace is captured for every allocation when it is created, just in case it leaks. This incurs
  some memory overhead to store data that is almost never used. This flag is implied by
  `-Zmiri-ignore-leaks`.
* `-Zmiri-env-forward=<var>` forwards the `var` environment variable to the interpreted program. Can
  be used multiple times to forward several variables. Execution will still be deterministic if the
  value of forwarded variables stays the same. Has no effect if `-Zmiri-disable-isolation` is set.
* `-Zmiri-ignore-leaks` disables the memory leak checker, and also allows some
  remaining threads to exist when the main thread exits.
* `-Zmiri-isolation-error=<action>` configures Miri's response to operations
  requiring host access while isolation is enabled. `abort`, `hide`, `warn`,
  and `warn-nobacktrace` are the supported actions. The default is to `abort`,
  which halts the machine. Some (but not all) operations also support continuing
  execution with a "permission denied" error being returned to the program.
  `warn` prints a full backtrace when that happens; `warn-nobacktrace` is less
  verbose. `hide` hides the warning entirely.
* `-Zmiri-num-cpus` states the number of available CPUs to be reported by miri. By default, the
  number of available CPUs is `1`. Note that this flag does not affect how miri handles threads in
  any way.
* `-Zmiri-permissive-provenance` disables the warning for integer-to-pointer casts and
  [`ptr::from_exposed_addr`](https://doc.rust-lang.org/nightly/std/ptr/fn.from_exposed_addr.html).
  This will necessarily miss some bugs as those operations are not efficiently and accurately
  implementable in a sanitizer, but it will only miss bugs that concern memory/pointers which is
  subject to these operations.
* `-Zmiri-preemption-rate` configures the probability that at the end of a basic block, the active
  thread will be preempted. The default is `0.01` (i.e., 1%). Setting this to `0` disables
  preemption.
* `-Zmiri-report-progress` makes Miri print the current stacktrace every now and then, so you can
  tell what it is doing when a program just keeps running. You can customize how frequently the
  report is printed via `-Zmiri-report-progress=<blocks>`, which prints the report every N basic
  blocks.
* `-Zmiri-seed=<num>` configures the seed of the RNG that Miri uses to resolve non-determinism. This
  RNG is used to pick base addresses for allocations, to determine preemption and failure of
  `compare_exchange_weak`, and to control store buffering for weak memory emulation. When isolation
  is enabled (the default), this is also used to emulate system entropy. The default seed is 0. You
  can increase test coverage by running Miri multiple times with different seeds.
* `-Zmiri-strict-provenance` enables [strict
  provenance](https://github.com/rust-lang/rust/issues/95228) checking in Miri. This means that
  casting an integer to a pointer yields a result with 'invalid' provenance, i.e., with provenance
  that cannot be used for any memory access.
* `-Zmiri-symbolic-alignment-check` makes the alignment check more strict.  By default, alignment is
  checked by casting the pointer to an integer, and making sure that is a multiple of the alignment.
  This can lead to cases where a program passes the alignment check by pure chance, because things
  "happened to be" sufficiently aligned -- there is no UB in this execution but there would be UB in
  others.  To avoid such cases, the symbolic alignment check only takes into account the requested
  alignment of the relevant allocation, and the offset into that allocation.  This avoids missing
  such bugs, but it also incurs some false positives when the code does manual integer arithmetic to
  ensure alignment.  (The standard library `align_to` method works fine in both modes; under
  symbolic alignment it only fills the middle slice when the allocation guarantees sufficient
  alignment.)

The remaining flags are for advanced use only, and more likely to change or be removed.
Some of these are **unsound**, which means they can lead
to Miri failing to detect cases of undefined behavior in a program.

* `-Zmiri-disable-abi-check` disables checking [function ABI]. Using this flag
  is **unsound**.
* `-Zmiri-disable-alignment-check` disables checking pointer alignment, so you
  can focus on other failures, but it means Miri can miss bugs in your program.
  Using this flag is **unsound**.
* `-Zmiri-disable-data-race-detector` disables checking for data races.  Using
  this flag is **unsound**. This implies `-Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-disable-stacked-borrows` disables checking the experimental
  aliasing rules to track borrows ([Stacked Borrows] and [Tree Borrows]).
  This can make Miri run faster, but it also means no aliasing violations will
  be detected. Using this flag is **unsound** (but the affected soundness rules
  are experimental). Later flags take precedence: borrow tracking can be reactivated
  by `-Zmiri-tree-borrows`.
* `-Zmiri-disable-validation` disables enforcing validity invariants, which are
  enforced by default.  This is mostly useful to focus on other failures (such
  as out-of-bounds accesses) first.  Setting this flag means Miri can miss bugs
  in your program.  However, this can also help to make Miri run faster.  Using
  this flag is **unsound**.
* `-Zmiri-disable-weak-memory-emulation` disables the emulation of some C++11 weak
  memory effects.
* `-Zmiri-extern-so-file=<path to a shared object file>` is an experimental flag for providing support
  for FFI calls. Functions not provided by that file are still executed via the usual Miri shims.
  **WARNING**: If an invalid/incorrect `.so` file is specified, this can cause undefined behaviour in Miri itself!
  And of course, Miri cannot do any checks on the actions taken by the external code.
  Note that Miri has its own handling of file descriptors, so if you want to replace *some* functions
  working on file descriptors, you will have to replace *all* of them, or the two kinds of
  file descriptors will be mixed up.
  This is **work in progress**; currently, only integer arguments and return values are
  supported (and no, pointer/integer casts to work around this limitation will not work;
  they will fail horribly). It also only works on unix hosts for now.
  Follow [the discussion on supporting other types](https://github.com/rust-lang/miri/issues/2365).
* `-Zmiri-measureme=<name>` enables `measureme` profiling for the interpreted program.
   This can be used to find which parts of your program are executing slowly under Miri.
   The profile is written out to a file inside a directory called `<name>`, and can be processed
   using the tools in the repository https://github.com/rust-lang/measureme.
* `-Zmiri-mute-stdout-stderr` silently ignores all writes to stdout and stderr,
  but reports to the program that it did actually write. This is useful when you
  are not interested in the actual program's output, but only want to see Miri's
  errors and warnings.
* `-Zmiri-panic-on-unsupported` will makes some forms of unsupported functionality,
  such as FFI and unsupported syscalls, panic within the context of the emulated
  application instead of raising an error within the context of Miri (and halting
  execution). Note that code might not expect these operations to ever panic, so
  this flag can lead to strange (mis)behavior.
* `-Zmiri-retag-fields` changes Stacked Borrows retagging to recurse into *all* fields.
  This means that references in fields of structs/enums/tuples/arrays/... are retagged,
  and in particular, they are protected when passed as function arguments.
  (The default is to recurse only in cases where rustc would actually emit a `noalias` attribute.)
* `-Zmiri-retag-fields=<all|none|scalar>` controls when Stacked Borrows retagging recurses into
  fields. `all` means it always recurses (like `-Zmiri-retag-fields`), `none` means it never
  recurses, `scalar` (the default) means it only recurses for types where we would also emit
  `noalias` annotations in the generated LLVM IR (types passed as individual scalars or pairs of
  scalars). Setting this to `none` is **unsound**.
* `-Zmiri-tag-gc=<blocks>` configures how often the pointer tag garbage collector runs. The default
  is to search for and remove unreachable tags once every `10000` basic blocks. Setting this to
  `0` disables the garbage collector, which causes some programs to have explosive memory usage
  and/or super-linear runtime.
* `-Zmiri-track-alloc-id=<id1>,<id2>,...` shows a backtrace when the given allocations are
  being allocated or freed.  This helps in debugging memory leaks and
  use after free bugs. Specifying this argument multiple times does not overwrite the previous
  values, instead it appends its values to the list. Listing an id multiple times has no effect.
* `-Zmiri-track-call-id=<id1>,<id2>,...` shows a backtrace when the given call ids are
  assigned to a stack frame.  This helps in debugging UB related to Stacked
  Borrows "protectors". Specifying this argument multiple times does not overwrite the previous
  values, instead it appends its values to the list. Listing an id multiple times has no effect.
* `-Zmiri-track-pointer-tag=<tag1>,<tag2>,...` shows a backtrace when a given pointer tag
  is created and when (if ever) it is popped from a borrow stack (which is where the tag becomes invalid
  and any future use of it will error).  This helps you in finding out why UB is
  happening and where in your code would be a good place to look for it.
  Specifying this argument multiple times does not overwrite the previous
  values, instead it appends its values to the list. Listing a tag multiple times has no effect.
* `-Zmiri-track-weak-memory-loads` shows a backtrace when weak memory emulation returns an outdated
  value from a load. This can help diagnose problems that disappear under
  `-Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-tree-borrows` replaces [Stacked Borrows] with the [Tree Borrows] rules.
  The soundness rules are already experimental without this flag, but even more
  so with this flag.
* `-Zmiri-force-page-size=<num>` overrides the default page size for an architecture, in multiples of 1k.
  `4` is default for most targets. This value should always be a power of 2 and nonzero.
* `-Zmiri-unique-is-unique` performs additional aliasing checks for `core::ptr::Unique` to ensure
  that it could theoretically be considered `noalias`. This flag is experimental and has
  an effect only when used with `-Zmiri-tree-borrows`.

[function ABI]: https://doc.rust-lang.org/reference/items/functions.html#extern-function-qualifier

Some native rustc `-Z` flags are also very relevant for Miri:

* `-Zmir-opt-level` controls how many MIR optimizations are performed.  Miri
  overrides the default to be `0`; be advised that using any higher level can
  make Miri miss bugs in your program because they got optimized away.
* `-Zalways-encode-mir` makes rustc dump MIR even for completely monomorphic
  functions.  This is needed so that Miri can execute such functions, so Miri
  sets this flag per default.
* `-Zmir-emit-retag` controls whether `Retag` statements are emitted. Miri
  enables this per default because it is needed for [Stacked Borrows] and [Tree Borrows].

Moreover, Miri recognizes some environment variables:

* `MIRI_AUTO_OPS` indicates whether the automatic execution of rustfmt, clippy and toolchain setup
  should be skipped. If it is set to any value, they are skipped. This is used for avoiding infinite
  recursion in `./miri` and to allow automated IDE actions to avoid the auto ops.
* `MIRI_LOG`, `MIRI_BACKTRACE` control logging and backtrace printing during
  Miri executions, also [see "Testing the Miri driver" in `CONTRIBUTING.md`][testing-miri].
* `MIRIFLAGS` (recognized by `cargo miri` and the test suite) defines extra
  flags to be passed to Miri.
* `MIRI_LIB_SRC` defines the directory where Miri expects the sources of the
  standard library that it will build and use for interpretation. This directory
  must point to the `library` subdirectory of a `rust-lang/rust` repository
  checkout. Note that changing files in that directory does not automatically
  trigger a re-build of the standard library; you have to clear the Miri build
  cache manually (on Linux, `rm -rf ~/.cache/miri`;
  on Windows, `rmdir /S "%LOCALAPPDATA%\rust-lang\miri\cache"`;
  and on macOS, `rm -rf ~/Library/Caches/org.rust-lang.miri`).
* `MIRI_SYSROOT` (recognized by `cargo miri` and the Miri driver) indicates the sysroot to use. When
  using `cargo miri`, this skips the automatic setup -- only set this if you do not want to use the
  automatically created sysroot. For directly invoking the Miri driver, this variable (or a
  `--sysroot` flag) is mandatory. When invoking `cargo miri setup`, this indicates where the sysroot
  will be put.
* `MIRI_TEST_TARGET` (recognized by the test suite and the `./miri` script) indicates which target
  architecture to test against.  `miri` and `cargo miri` accept the `--target` flag for the same
  purpose.
* `MIRI_NO_STD` (recognized by `cargo miri` and the test suite) makes sure that the target's
  sysroot is built without libstd. This allows testing and running no_std programs.
* `MIRI_BLESS` (recognized by the test suite and `cargo-miri-test/run-test.py`): overwrite all
  `stderr` and `stdout` files instead of checking whether the output matches.
* `MIRI_SKIP_UI_CHECKS` (recognized by the test suite): don't check whether the
  `stderr` or `stdout` files match the actual output.

The following environment variables are *internal* and must not be used by
anyone but Miri itself. They are used to communicate between different Miri
binaries, and as such worth documenting:

* `MIRI_BE_RUSTC` can be set to `host` or `target`. It tells the Miri driver to
  actually not interpret the code but compile it like rustc would. With `target`, Miri sets
  some compiler flags to prepare the code for interpretation; with `host`, this is not done.
  This environment variable is useful to be sure that the compiled `rlib`s are compatible
  with Miri.
* `MIRI_CALLED_FROM_SETUP` is set during the Miri sysroot build,
  which will re-invoke `cargo-miri` as the `rustc` to use for this build.
* `MIRI_CALLED_FROM_RUSTDOC` when set to any value tells `cargo-miri` that it is
  running as a child process of `rustdoc`, which invokes it twice for each doc-test
  and requires special treatment, most notably a check-only build before interpretation.
  This is set by `cargo-miri` itself when running as a `rustdoc`-wrapper.
* `MIRI_CWD` when set to any value tells the Miri driver to change to the given
  directory after loading all the source files, but before commencing
  interpretation. This is useful if the interpreted program wants a different
  working directory at run-time than at build-time.
* `MIRI_LOCAL_CRATES` is set by `cargo-miri` to tell the Miri driver which
  crates should be given special treatment in diagnostics, in addition to the
  crate currently being compiled.
* `MIRI_VERBOSE` when set to any value tells the various `cargo-miri` phases to
  perform verbose logging.
* `MIRI_HOST_SYSROOT` is set by bootstrap to tell `cargo-miri` which sysroot to use for *host*
  operations.

[testing-miri]: CONTRIBUTING.md#testing-the-miri-driver

## Miri `extern` functions

Miri provides some `extern` functions that programs can import to access
Miri-specific functionality. They are declared in
[/tests/utils/miri\_extern.rs](/tests/utils/miri_extern.rs).

## Contributing and getting help

If you want to contribute to Miri, great!  Please check out our
[contribution guide](CONTRIBUTING.md).

For help with running Miri, you can open an issue here on
GitHub or use the [Miri stream on the Rust Zulip][zulip].

[zulip]: https://rust-lang.zulipchat.com/#narrow/stream/269128-miri

## History

This project began as part of an undergraduate research course in 2015 by
@solson at the [University of Saskatchewan][usask].  There are [slides] and a
[report] available from that project.  In 2016, @oli-obk joined to prepare Miri
for eventually being used as const evaluator in the Rust compiler itself
(basically, for `const` and `static` stuff), replacing the old evaluator that
worked directly on the AST.  In 2017, @RalfJung did an internship with Mozilla
and began developing Miri towards a tool for detecting undefined behavior, and
also using Miri as a way to explore the consequences of various possible
definitions for undefined behavior in Rust.  @oli-obk's move of the Miri engine
into the compiler finally came to completion in early 2018.  Meanwhile, later
that year, @RalfJung did a second internship, developing Miri further with
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
* [TiKV using `Vec::from_raw_parts` incorrectly](https://github.com/tikv/agatedb/pull/24)
* Incorrect doctests for [`AtomicPtr`](https://github.com/rust-lang/rust/pull/84052) and [`Box::from_raw_in`](https://github.com/rust-lang/rust/pull/84053)
* [Insufficient alignment in `ThinVec`](https://github.com/Gankra/thin-vec/pull/27)
* [`crossbeam-epoch` calling `assume_init` on a partly-initialized `MaybeUninit`](https://github.com/crossbeam-rs/crossbeam/pull/779)
* [`integer-encoding` dereferencing a misaligned pointer](https://github.com/dermesser/integer-encoding-rs/pull/23)
* [`rkyv` constructing a `Box<[u8]>` from an overaligned allocation](https://github.com/rkyv/rkyv/commit/a9417193a34757e12e24263178be8b2eebb72456)
* [Data race in `arc-swap`](https://github.com/vorner/arc-swap/issues/76)
* [Data race in `thread::scope`](https://github.com/rust-lang/rust/issues/98498)
* [`regex` incorrectly handling unaligned `Vec<u8>` buffers](https://www.reddit.com/r/rust/comments/vq3mmu/comment/ienc7t0?context=3)
* [Incorrect use of `compare_exchange_weak` in `once_cell`](https://github.com/matklad/once_cell/issues/186)
* [Dropping with unaligned pointers in `vec::IntoIter`](https://github.com/rust-lang/rust/pull/106084)

Violations of [Stacked Borrows] found that are likely bugs (but Stacked Borrows is currently just an experiment):

* [`VecDeque::drain` creating overlapping mutable references](https://github.com/rust-lang/rust/pull/56161)
* Various `BTreeMap` problems
    * [`BTreeMap` iterators creating mutable references that overlap with shared references](https://github.com/rust-lang/rust/pull/58431)
    * [`BTreeMap::iter_mut` creating overlapping mutable references](https://github.com/rust-lang/rust/issues/73915)
    * [`BTreeMap` node insertion using raw pointers outside their valid memory area](https://github.com/rust-lang/rust/issues/78477)
* [`LinkedList` cursor insertion creating overlapping mutable references](https://github.com/rust-lang/rust/pull/60072)
* [`Vec::push` invalidating existing references into the vector](https://github.com/rust-lang/rust/issues/60847)
* [`align_to_mut` violating uniqueness of mutable references](https://github.com/rust-lang/rust/issues/68549)
* [`sized-chunks` creating aliasing mutable references](https://github.com/bodil/sized-chunks/issues/8)
* [`String::push_str` invalidating existing references into the string](https://github.com/rust-lang/rust/issues/70301)
* [`ryu` using raw pointers outside their valid memory area](https://github.com/dtolnay/ryu/issues/24)
* [ink! creating overlapping mutable references](https://github.com/rust-lang/miri/issues/1364)
* [TiKV creating overlapping mutable reference and raw pointer](https://github.com/tikv/tikv/pull/7709)
* [Windows `Env` iterator using a raw pointer outside its valid memory area](https://github.com/rust-lang/rust/pull/70479)
* [`VecDeque::iter_mut` creating overlapping mutable references](https://github.com/rust-lang/rust/issues/74029)
* [Various standard library aliasing issues involving raw pointers](https://github.com/rust-lang/rust/pull/78602)
* [`<[T]>::copy_within` using a loan after invalidating it](https://github.com/rust-lang/rust/pull/85610)

## Scientific papers employing Miri

* [Stacked Borrows: An Aliasing Model for Rust](https://plv.mpi-sws.org/rustbelt/stacked-borrows/)
* [Using Lightweight Formal Methods to Validate a Key-Value Storage Node in Amazon S3](https://www.amazon.science/publications/using-lightweight-formal-methods-to-validate-a-key-value-storage-node-in-amazon-s3)
* [SyRust: Automatic Testing of Rust Libraries with Semantic-Aware Program Synthesis](https://dl.acm.org/doi/10.1145/3453483.3454084)

## License

Licensed under either of

  * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or
    http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
