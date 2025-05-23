# Miri

Miri is an [Undefined Behavior][reference-ub] detection tool for Rust. It can run binaries and test
suites of cargo projects and detect unsafe code that fails to uphold its safety requirements. For
instance:

* Out-of-bounds memory accesses and use-after-free
* Invalid use of uninitialized data
* Violation of intrinsic preconditions (an [`unreachable_unchecked`] being
  reached, calling [`copy_nonoverlapping`] with overlapping ranges, ...)
* Not sufficiently aligned memory accesses and references
* Violation of basic type invariants (a `bool` that is not 0 or 1, for example,
  or an invalid enum discriminant)
* **Experimental**: Violations of the [Stacked Borrows] rules governing aliasing
  for reference types
* **Experimental**: Violations of the [Tree Borrows] aliasing rules, as an optional
  alternative to [Stacked Borrows]
* **Experimental**: Data races and emulation of weak memory effects, i.e.,
  atomic reads can return outdated values.

On top of that, Miri will also tell you about memory leaks: when there is memory
still allocated at the end of the execution, and that memory is not reachable
from a global `static`, Miri will raise an error.

You can use Miri to emulate programs on other targets, e.g. to ensure that
byte-level data manipulation works correctly both on little-endian and
big-endian systems. See
[cross-interpretation](#cross-interpretation-running-for-different-targets)
below.

Miri has already discovered many [real-world bugs](#bugs-found-by-miri). If you
found a bug with Miri, we'd appreciate if you tell us and we'll add it to the
list!

By default, Miri ensures a fully deterministic execution and isolates the
program from the host system. Some APIs that would usually access the host, such
as gathering entropy for random number generators, environment variables, and
clocks, are replaced by deterministic "fake" implementations. Set
`MIRIFLAGS="-Zmiri-disable-isolation"` to access the real system APIs instead.
(In particular, the "fake" system RNG APIs make Miri **not suited for
cryptographic use**! Do not generate keys using Miri.)

All that said, be aware that Miri does **not catch every violation of the Rust specification** in
your program, not least because there is no such specification. Miri uses its own approximation of
what is and is not Undefined Behavior in Rust. To the best of our knowledge, all Undefined Behavior
that has the potential to affect a program's correctness *is* being detected by Miri (modulo
[bugs][I-misses-ub]), but you should consult [the Reference][reference-ub] for the official
definition of Undefined Behavior. Miri will be updated with the Rust compiler to protect against UB
as it is understood by the current compiler, but it makes no promises about future versions of
rustc.

Further caveats that Miri users should be aware of:

* If the program relies on unspecified details of how data is laid out, it will
  still run fine in Miri -- but might break (including causing UB) on different
  compiler versions or different platforms. (You can use `-Zrandomize-layout`
  to detect some of these cases.)
* Program execution is non-deterministic when it depends, for example, on where
  exactly in memory allocations end up, or on the exact interleaving of
  concurrent threads. Miri tests one of many possible executions of your
  program, but it will miss bugs that only occur in a different possible execution.
  You can alleviate this to some extent by running Miri with different
  values for `-Zmiri-seed`, but that will still by far not explore all possible executions.
* Miri runs the program as a platform-independent interpreter, so the program
  has no access to most platform-specific APIs or FFI. A few APIs have been
  implemented (such as printing to stdout, accessing environment variables, and
  basic file system access) but most have not: for example, Miri currently does
  not support networking. System API support varies between targets; if you run
  on Windows it is a good idea to use `--target x86_64-unknown-linux-gnu` to get
  better support.
* Weak memory emulation is not complete: there are legal behaviors that Miri will never produce.
  However, Miri produces many behaviors that are hard to observe on real hardware, so it can help
  quite a bit in finding weak memory concurrency bugs. To be really sure about complicated atomic
  code, use specialized tools such as [loom](https://github.com/tokio-rs/loom).

Moreover, Miri fundamentally cannot ensure that your code is *sound*. [Soundness] is the property of
never causing undefined behavior when invoked from arbitrary safe code, even in combination with
other sound code. In contrast, Miri can just tell you if *a particular way of interacting with your
code* (e.g., a test suite) causes any undefined behavior *in a particular execution* (of which there
may be many, e.g. when concurrency or other forms of non-determinism are involved). When Miri finds
UB, your code is definitely unsound, but when Miri does not find UB, then you may just have to test
more inputs or more possible non-deterministic choices.

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[`unreachable_unchecked`]: https://doc.rust-lang.org/stable/std/hint/fn.unreachable_unchecked.html
[`copy_nonoverlapping`]: https://doc.rust-lang.org/stable/std/ptr/fn.copy_nonoverlapping.html
[Stacked Borrows]: https://github.com/rust-lang/unsafe-code-guidelines/blob/master/wip/stacked-borrows.md
[Tree Borrows]: https://perso.crans.org/vanille/treebor/
[Soundness]: https://rust-lang.github.io/unsafe-code-guidelines/glossary.html#soundness-of-code--of-a-library
[reference-ub]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
[I-misses-ub]: https://github.com/rust-lang/miri/labels/I-misses-UB


## Using Miri

Install Miri on Rust nightly via `rustup`:

```sh
rustup +nightly component add miri
```

All the following commands assume the nightly toolchain is pinned via `rustup override set nightly`.
Alternatively, use `cargo +nightly` for each of the following commands.

Now you can run your project in Miri:

- To run all tests in your project through Miri, use `cargo miri test`.
- If you have a binary project, you can run it through Miri using `cargo miri run`.

The first time you run Miri, it will perform some extra setup and install some
dependencies. It will ask you for confirmation before installing anything.

`cargo miri run/test` supports the exact same flags as `cargo run/test`. For
example, `cargo miri test filter` only runs the tests containing `filter` in
their name.

You can pass [flags][miri-flags] to Miri via `MIRIFLAGS`. For example,
`MIRIFLAGS="-Zmiri-disable-stacked-borrows" cargo miri run` runs the program
without checking the aliasing of references.

When compiling code via `cargo miri`, the `cfg(miri)` config flag is set for code
that will be interpreted under Miri. You can use this to ignore test cases that fail
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
            performed an operation that Miri does not support
```

### Cross-interpretation: running for different targets

Miri can not only run a binary or test suite for your host target, it can also
perform cross-interpretation for arbitrary foreign targets: `cargo miri run
--target x86_64-unknown-linux-gnu` will run your program as if it was a Linux
program, no matter your host OS. This is particularly useful if you are using
Windows, as the Linux target is much better supported than Windows targets.

You can also use this to test platforms with different properties than your host
platform. For example `cargo miri test --target s390x-unknown-linux-gnu`
will run your test suite on a big-endian target, which is useful for testing
endian-sensitive code.

### Testing multiple different executions

Certain parts of the execution are picked randomly by Miri, such as the exact base address
allocations are stored at and the interleaving of concurrently executing threads. Sometimes, it can
be useful to explore multiple different execution, e.g. to make sure that your code does not depend
on incidental "super-alignment" of new allocations and to test different thread interleavings.
This can be done with the `-Zmiri-many-seeds` flag:

```
MIRIFLAGS="-Zmiri-many-seeds" cargo miri test # tries the seeds in 0..64
MIRIFLAGS="-Zmiri-many-seeds=0..16" cargo miri test
```

The default of 64 different seeds can be quite slow, so you often want to specify a smaller range.

### Running Miri on CI

When running Miri on CI, use the following snippet to install a nightly toolchain with the Miri
component:

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
      - uses: actions/checkout@v4
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

### Supported targets

Miri does not support all targets supported by Rust. The good news, however, is
that no matter your host OS/platform, it is easy to run code for *any* target
using `--target`!

The following targets are tested on CI and thus should always work (to the
degree documented below):

- All Rust [Tier 1 targets](https://doc.rust-lang.org/rustc/platform-support.html) are supported by
  Miri. They are all checked on Miri's CI, and some (at least one per OS) are even checked on every
  Rust PR, so the shipped Miri should always work on these targets.
- `s390x-unknown-linux-gnu` is supported as our "big-endian target of choice".
- For every other target with OS `linux`, `macos`, or `windows`, Miri should generally work, but we
  make no promises and we don't run tests for such targets.
- We have unofficial support (not maintained by the Miri team itself) for some further operating systems.
  - `solaris` / `illumos`: maintained by @devnexen. Supports the entire test suite.
  - `freebsd`: maintained by @YohDeadfall and @LorrensP-2158466. Supports the entire test suite.
  - `android`: **maintainer wanted**. Support very incomplete, but a basic "hello world" works.
  - `wasi`: **maintainer wanted**. Support very incomplete, not even standard output works, but an empty `main` function works.
- For targets on other operating systems, Miri might fail before even reaching the `main` function.

However, even for targets that we do support, the degree of support for accessing platform APIs
(such as the file system) differs between targets: generally, Linux targets have the best support,
and macOS targets are usually on par. Windows is supported less well.

### Running tests in parallel

Though it implements Rust threading, Miri itself is a single-threaded interpreter.
This means that when running `cargo miri test`, you will probably see a dramatic
increase in the amount of time it takes to run your whole test suite due to the
inherent interpreter slowdown and a loss of parallelism.

You can get your test suite's parallelism back by running `cargo miri nextest run -jN`
(note that you will need [`cargo-nextest`](https://nexte.st) installed).
This works because `cargo-nextest` collects a list of all tests then launches a
separate `cargo miri run` for each test. For more information about nextest, see the
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

#### "found crate `std` compiled by an incompatible version of rustc"

You may be running `cargo miri` with a different compiler version than the one
used to build the custom libstd that Miri uses, and Miri failed to detect that.
Try running `cargo miri clean`.


## Miri `-Z` flags and environment variables
[miri-flags]: #miri--z-flags-and-environment-variables

Miri adds its own set of `-Z` flags, which are usually set via the `MIRIFLAGS`
environment variable. We first document the most relevant and most commonly used flags:

* `-Zmiri-deterministic-concurrency` makes Miri's concurrency-related behavior fully deterministic.
  Strictly speaking, Miri is always fully deterministic when isolation is enabled (the default
  mode), but this determinism is achieved by using an RNG with a fixed seed. Seemingly harmless
  changes to the program, or just running it for a different target architecture, can thus lead to
  completely different program behavior down the line. This flag disables the use of an RNG for
  concurrency-related decisions. Therefore, Miri cannot find bugs that only occur under some
  specific circumstances, but Miri's behavior will also be more stable across versions and targets.
  This is equivalent to `-Zmiri-fixed-schedule -Zmiri-compare-exchange-weak-failure-rate=0.0
  -Zmiri-address-reuse-cross-thread-rate=0.0 -Zmiri-disable-weak-memory-emulation`.
* `-Zmiri-disable-isolation` disables host isolation. As a consequence,
  the program has access to host resources such as environment variables, file
  systems, and randomness.
  This overwrites a previous `-Zmiri-isolation-error`.
* `-Zmiri-disable-leak-backtraces` disables backtraces reports for memory leaks. By default, a
  backtrace is captured for every allocation when it is created, just in case it leaks. This incurs
  some memory overhead to store data that is almost never used. This flag is implied by
  `-Zmiri-ignore-leaks`.
* `-Zmiri-env-forward=<var>` forwards the `var` environment variable to the interpreted program. Can
  be used multiple times to forward several variables. Execution will still be deterministic if the
  value of forwarded variables stays the same. Has no effect if `-Zmiri-disable-isolation` is set.
* `-Zmiri-env-set=<var>=<value>` sets the `var` environment variable to `value` in the interpreted program.
  It can be used to pass environment variables without needing to alter the host environment. It can
  be used multiple times to set several variables. If `-Zmiri-disable-isolation` or `-Zmiri-env-forward`
  is set, values set with this option will have priority over values from the host environment.
* `-Zmiri-ignore-leaks` disables the memory leak checker, and also allows some
  remaining threads to exist when the main thread exits.
* `-Zmiri-isolation-error=<action>` configures Miri's response to operations
  requiring host access while isolation is enabled. `abort`, `hide`, `warn`,
  and `warn-nobacktrace` are the supported actions. The default is to `abort`,
  which halts the machine. Some (but not all) operations also support continuing
  execution with a "permission denied" error being returned to the program.
  `warn` prints a full backtrace each time that happens; `warn-nobacktrace` is less
  verbose and shown at most once per operation. `hide` hides the warning entirely.
  This overwrites a previous `-Zmiri-disable-isolation`.
* `-Zmiri-many-seeds=[<from>]..<to>` runs the program multiple times with different seeds for Miri's
  RNG. With different seeds, Miri will make different choices to resolve non-determinism such as the
  order in which concurrent threads are scheduled, or the exact addresses assigned to allocations.
  This is useful to find bugs that only occur under particular interleavings of concurrent threads,
  or that otherwise depend on non-determinism. If the `<from>` part is skipped, it defaults to `0`.
  Can be used without a value; in that case the range defaults to `0..64`.
* `-Zmiri-many-seeds-keep-going` tells Miri to really try all the seeds in the given range, even if
  a failing seed has already been found. This is useful to determine which fraction of seeds fails.
* `-Zmiri-num-cpus` states the number of available CPUs to be reported by miri. By default, the
  number of available CPUs is `1`. Note that this flag does not affect how miri handles threads in
  any way.
* `-Zmiri-permissive-provenance` disables the warning for integer-to-pointer casts and
  [`ptr::with_exposed_provenance`](https://doc.rust-lang.org/nightly/std/ptr/fn.with_exposed_provenance.html).
  This will necessarily miss some bugs as those operations are not efficiently and accurately
  implementable in a sanitizer, but it will only miss bugs that concern memory/pointers which is
  subject to these operations.
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
  casting an integer to a pointer will stop execution because the provenance of the pointer
  cannot be determined.
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

* `-Zmiri-address-reuse-rate=<rate>` changes the probability that a freed *non-stack* allocation
  will be added to the pool for address reuse, and the probability that a new *non-stack* allocation
  will be taken from the pool. Stack allocations never get added to or taken from the pool. The
  default is `0.5`.
* `-Zmiri-address-reuse-cross-thread-rate=<rate>` changes the probability that an allocation which
  attempts to reuse a previously freed block of memory will also consider blocks freed by *other
  threads*. The default is `0.1`, which means by default, in 90% of the cases where an address reuse
  attempt is made, only addresses from the same thread will be considered. Reusing an address from
  another thread induces synchronization between those threads, which can mask data races and weak
  memory bugs.
* `-Zmiri-compare-exchange-weak-failure-rate=<rate>` changes the failure rate of
  `compare_exchange_weak` operations. The default is `0.8` (so 4 out of 5 weak ops will fail).
  You can change it to any value between `0.0` and `1.0`, where `1.0` means it
  will always fail and `0.0` means it will never fail. Note that setting it to
  `1.0` will likely cause hangs, since it means programs using
  `compare_exchange_weak` cannot make progress.
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
* `-Zmiri-fixed-schedule` disables preemption (like `-Zmiri-preemption-rate=0.0`) and furthermore
  disables the randomization of the next thread to be picked, instead fixing a round-robin schedule.
  Note however that other aspects of Miri's concurrency behavior are still randomize; use
  `-Zmiri-deterministic-concurrency` to disable them all.
* `-Zmiri-force-intrinsic-fallback` forces the use of the "fallback" body for all intrinsics that
  have one. This is useful to test the fallback bodies, but should not be used otherwise. It is
  **unsound** since the fallback body might not be checking for all UB.
* `-Zmiri-native-lib=<path to a shared object file>` is an experimental flag for providing support
  for calling native functions from inside the interpreter via FFI. The flag is supported only on
  Unix systems. Functions not provided by that file are still executed via the usual Miri shims.
  **WARNING**: If an invalid/incorrect `.so` file is specified, this can cause Undefined Behavior in
  Miri itself! And of course, Miri cannot do any checks on the actions taken by the native code.
  Note that Miri has its own handling of file descriptors, so if you want to replace *some*
  functions working on file descriptors, you will have to replace *all* of them, or the two kinds of
  file descriptors will be mixed up. This is **work in progress**; currently, only integer and
  pointers arguments and return values are supported and memory allocated by the native code cannot
  be accessed from Rust (only the other way around). Native code must not spawn threads that keep
  running in the background after the call has returned to Rust and that access Rust-allocated
  memory. Finally, the flag is **unsound** in the sense that Miri stops tracking details such as
  initialization and provenance on memory shared with native code, so it is easily possible to write
  code that has UB which is missed by Miri.
* `-Zmiri-measureme=<name>` enables `measureme` profiling for the interpreted program.
   This can be used to find which parts of your program are executing slowly under Miri.
   The profile is written out to a file inside a directory called `<name>`, and can be processed
   using the tools in the repository https://github.com/rust-lang/measureme.
* `-Zmiri-mute-stdout-stderr` silently ignores all writes to stdout and stderr,
  but reports to the program that it did actually write. This is useful when you
  are not interested in the actual program's output, but only want to see Miri's
  errors and warnings.
* `-Zmiri-recursive-validation` is a *highly experimental* flag that makes validity checking
  recurse below references.
* `-Zmiri-retag-fields[=<all|none|scalar>]` controls when Stacked Borrows retagging recurses into
  fields. `all` means it always recurses (the default, and equivalent to `-Zmiri-retag-fields`
  without an explicit value), `none` means it never recurses, `scalar` means it only recurses for
  types where we would also emit `noalias` annotations in the generated LLVM IR (types passed as
  individual scalars or pairs of scalars). Setting this to `none` is **unsound**.
* `-Zmiri-preemption-rate` configures the probability that at the end of a basic block, the active
  thread will be preempted. The default is `0.01` (i.e., 1%). Setting this to `0` disables
  preemption. Note that even without preemption, the schedule is still non-deterministic:
  if a thread blocks or yields, the next thread is chosen randomly.
* `-Zmiri-provenance-gc=<blocks>` configures how often the pointer provenance garbage collector runs.
  The default is to search for and remove unreachable provenance once every `10000` basic blocks. Setting
  this to `0` disables the garbage collector, which causes some programs to have explosive memory
  usage and/or super-linear runtime.
* `-Zmiri-track-alloc-accesses` show not only allocation and free events for tracked allocations,
  but also reads and writes.
* `-Zmiri-track-alloc-id=<id1>,<id2>,...` shows a backtrace when the given allocations are
  being allocated or freed.  This helps in debugging memory leaks and
  use after free bugs. Specifying this argument multiple times does not overwrite the previous
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
* <a name="-Zmiri-tree-borrows"><!-- The playground links here --></a>
  `-Zmiri-tree-borrows` replaces [Stacked Borrows] with the [Tree Borrows] rules.
  Tree Borrows is even more experimental than Stacked Borrows. While Tree Borrows
  is still sound in the sense of catching all aliasing violations that current versions
  of the compiler might exploit, it is likely that the eventual final aliasing model
  of Rust will be stricter than Tree Borrows. In other words, if you use Tree Borrows,
  even if your code is accepted today, it might be declared UB in the future.
  This is much less likely with Stacked Borrows.
  Using Tree Borrows currently implies `-Zmiri-strict-provenance` because integer-to-pointer
  casts are not supported in this mode, but that may change in the future.
* `-Zmiri-force-page-size=<num>` overrides the default page size for an architecture, in multiples of 1k.
  `4` is default for most targets. This value should always be a power of 2 and nonzero.

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

* `MIRIFLAGS` defines extra flags to be passed to Miri.
* `MIRI_LIB_SRC` defines the directory where Miri expects the sources of the standard library that
  it will build and use for interpretation. This directory must point to the `library` subdirectory
  of a `rust-lang/rust` repository checkout.
* `MIRI_SYSROOT` indicates the sysroot to use. When using `cargo miri test`/`cargo miri run`, this skips the automatic
  setup -- only set this if you do not want to use the automatically created sysroot. When invoking
  `cargo miri setup`, this indicates where the sysroot will be put.
* `MIRI_NO_STD` makes sure that the target's sysroot is built without libstd. This allows testing
  and running no_std programs. This should *not usually be used*; Miri has a heuristic to detect
  no-std targets based on the target name. Setting this on a target that does support libstd can
  lead to confusing results.

[testing-miri]: CONTRIBUTING.md#testing-the-miri-driver

## Miri `extern` functions

Miri provides some `extern` functions that programs can import to access
Miri-specific functionality. They are declared in
[/tests/utils/miri\_extern.rs](/tests/utils/miri_extern.rs).

## Entry point for no-std binaries

Binaries that do not use the standard library are expected to declare a function like this so that
Miri knows where it is supposed to start execution:

```rust
#[cfg(miri)]
#[unsafe(no_mangle)]
fn miri_start(argc: isize, argv: *const *const u8) -> isize {
    // Call the actual start function that your project implements, based on your target's conventions.
}
```

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

Miri has already found a number of bugs in the Rust standard library and beyond, some of which we collect here.
If Miri helped you find a subtle UB bug in your code, we'd appreciate a PR adding it to the list!

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
* [Deallocating with the wrong layout in new specializations for in-place `Iterator::collect`](https://github.com/rust-lang/rust/pull/118460)
* [Incorrect offset computation for highly-aligned types in `portable-atomic-util`](https://github.com/taiki-e/portable-atomic/pull/138)
* [Occasional memory leak in `std::mpsc` channels](https://github.com/rust-lang/rust/issues/121582) (original code in [crossbeam](https://github.com/crossbeam-rs/crossbeam/pull/1084))
* [Weak-memory-induced memory leak in Windows thread-local storage](https://github.com/rust-lang/rust/pull/124281)
* [A bug in the new `RwLock::downgrade` implementation](https://rust-lang.zulipchat.com/#narrow/channel/269128-miri/topic/Miri.20error.20library.20test) (caught by Miri before it landed in the Rust repo)
* [Mockall reading unintialized memory when mocking `std::io::Read::read`, even if all expectations are satisfied](https://github.com/asomers/mockall/issues/647) (caught by Miri running Tokio's test suite)
* [`ReentrantLock` not correctly dealing with reuse of addresses for TLS storage of different threads](https://github.com/rust-lang/rust/pull/141248)

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
