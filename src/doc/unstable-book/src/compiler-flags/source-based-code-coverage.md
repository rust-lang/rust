# `source-based-code-coverage`

The feature request for this feature is: [#34701]

The Major Change Proposal (MCP) for this feature is: [#278](https://github.com/rust-lang/compiler-team/issues/278)

------------------------

## Introduction

The Rust compiler includes two code coverage implementations:

* A GCC-compatible, gcov-based coverage implementation, enabled with [`-Zprofile`](profile.md), which operates on DebugInfo.
* A source-based code coverage implementation, enabled with `-Zinstrument-coverage`, which uses LLVM's native coverage instrumentation to generate very precise coverage data.

This document describes how to enable and use the LLVM instrumentation-based coverage, via the `-Zinstrument-coverage` compiler flag.

## How it works

When `-Zinstrument-coverage` is enabled, the Rust compiler enhances rust-based libraries and binaries by:

* Automatically injecting calls to an LLVM intrinsic ([`llvm.instrprof.increment`]), at functions and branches in compiled code, to increment counters when conditional sections of code are executed.
* Embedding additional information in the data section of each library and binary (using the [LLVM Code Coverage Mapping Format]), to define the code regions (start and end positions in the source code) being counted.

When running a coverage-instrumented program, the counter values are written to a `profraw` file at program termination. LLVM bundles tools that read the counter results, combine those results with the coverage map (embedded in the program binary), and generate coverage reports in multiple formats.

## Enable coverage profiling in the Rust compiler

*IMPORTANT:* Rust's coverage profiling features may not be enabled, by default. To enable them, you may need to build a version of the Rust compiler with the `profiler` feature enabled.

First, edit the `config.toml` file, and find the `profiler` feature entry. Uncomment it and set it to `true`:

```toml
# Build the profiler runtime (required when compiling with options that depend
# on this runtime, such as `-C profile-generate` or `-Z instrument-coverage`).
profiler = true
```

Then rebuild the Rust compiler (see [rustc-dev-guide-how-to-build-and-run]).

### Building the demangler

LLVM coverage reporting tools generate results that can include function names and other symbol references, and the raw coverage results report symbols using the compiler's "mangled" version of the symbol names, which can be difficult to interpret. To work around this issue, LLVM coverage tools also support a user-specified symbol name demangler. Rust's symbol name demangler can be built with:

```shell
$ ./x.py build rust-demangler
```

## Compiling with coverage enabled

Set the `-Zinstrument-coverage` compiler flag in order to enable LLVM source-based code coverage profiling.

With `cargo`, you can instrument your program binary *and* dependencies at the same time.

For example (if your project's Cargo.toml builds a binary by default):

```shell
$ cd your-project
$ cargo clean
$ RUSTFLAGS="-Zinstrument-coverage" cargo build
```

If `cargo` is not configured to use your `profiler`-enabled version of `rustc`, set the path explicitly via the `RUSTC` environment variable. Here is another example, using a `stage1` build of `rustc` to compile an `example` binary (from the [`json5format`](https://crates.io/crates/json5format) crate):

```shell
$ RUSTC=$HOME/rust/build/x86_64-unknown-linux-gnu/stage1/bin/rustc \
    RUSTFLAGS="-Zinstrument-coverage" \
    cargo build --example formatjson5
```

## Running the instrumented binary to generate raw coverage profiling data

In the previous example, `cargo` generated the coverage-instrumented binary `formatjson5`:

```shell
$ echo "{some: 'thing'}" | target/debug/examples/formatjson5 -
```
```json5
{
    some: 'thing',
}
```

After running this program, a new file, `default.profraw`, should be in the current working directory. It's often preferable to set a specific file name or path. You can change the output file using the environment variable `LLVM_PROFILE_FILE`:


```shell
$ echo "{some: 'thing'}" \
    | LLVM_PROFILE_FILE="formatjson5.profraw" target/debug/examples/formatjson5 -
...
$ ls formatjson5.profraw
formatjson5.profraw
```

If `LLVM_PROFILE_FILE` contains a path to a non-existent directory, the missing directory structure will be created. Additionally, the following special pattern strings are rewritten:

* `%p` - The process ID.
* `%h` - The hostname of the machine running the program.
* `%t` - The value of the TMPDIR environment variable.
* `%Nm` - the instrumented binary’s signature: The runtime creates a pool of N raw profiles, used for on-line profile merging. The runtime takes care of selecting a raw profile from the pool, locking it, and updating it before the program exits. `N` must be between `1` and `9`, and defaults to `1` if omitted (with simply `%m`).
* `%c` - Does not add anything to the filename, but enables a mode (on some platforms, including Darwin) in which profile counter updates are continuously synced to a file. This means that if the instrumented program crashes, or is killed by a signal, perfect coverage information can still be recovered.

## Creating coverage reports

LLVM's tools to process coverage data and coverage maps have some version dependencies. If you encounter a version mismatch, try updating your LLVM tools, or use the LLVM tools bundled with the same Rust distrubition used to rebuild the Rust compiler (as shown in the following examples).

Raw profiles have to be indexed before they can be used to generate coverage reports. This is done using [`llvm-profdata merge`] (which can combine multiple raw profiles and index them at the same time):

```shell
$ $HOME/rust/build/x86_64-unknown-linux-gnu/llvm/bin/llvm-profdata merge \
    -sparse formatjson5.profraw -o formatjson5.profdata
```

Finally, the `.profdata` file is used, in combination with the coverage map (from the program binary) to generate coverage reports using [`llvm-cov report`]--for a coverage summaries--and [`llvm-cov show`]--to see detailed coverage of lines and regions (character ranges), overlaid on the original source code.

These commands have several display and filtering options. For example:

```shell
$ $HOME/rust/build/x86_64-unknown-linux-gnu/llvm/bin/llvm-cov show \
    -instr-profile=formatjson5.profdata \
    target/debug/examples/formatjson5 \
    -show-line-counts-or-regions \
    -Xdemangler=$HOME/rust/build/x86_64-unknown-linux-gnu/stage0-tools-bin/rust-demangler \
    -show-instantiations \
    -name=add_quoted_string
```

<img alt="Screenshot of sample `llvm-cov show` result, for function add_quoted_string" src="img/llvm-cov-show-01.png" class="center"/>
<br/>
<br/>

Some of the more notable options in this example include:

* `--instr-profile=<path-to-file>.profdata` - the location of the `.profdata` file created by `llvm-profdata merge`
* `target/debug/examples/formatjson5` - the binary that generated the coverage profiling data (originally as a `.profraw` file)
* `--Xdemangler=<path-to>/rust-demangler` - the location of the `rust-demangler` tool
* `--name=<exact-function-name>` - to show coverage for a specific function (or, consider using another filter option, such as `--name-regex=<pattern>`)

## Interpreting reports

There are four statistics tracked in a coverage summary:

* Function coverage is the percentage of functions that have been executed at least once. A function is considered to be executed if any of its instantiations are executed.
* Instantiation coverage is the percentage of function instantiations that have been executed at least once. Generic functions and functions generated from macros are two kinds of functions that may have multiple instantiations.
* Line coverage is the percentage of code lines that have been executed at least once. Only executable lines within function bodies are considered to be code lines.
* Region coverage is the percentage of code regions that have been executed at least once. A code region may span multiple lines: for example, in a large function body with no control flow. In other cases, a single line can contain multiple code regions: `return x || (y && z)` has countable code regions for `x` (which may resolve the expression, if `x` is `true`), `|| (y && z)` (executed only if `x` was `false`), and `return` (executed in either situation).

Of these four statistics, function coverage is usually the least granular while region coverage is the most granular. The project-wide totals for each statistic are listed in the summary.

## Other references

Rust's implementation and workflow for source-based code coverage is based on the same library and tools used to implement [source-based code coverage in Clang](https://clang.llvm.org/docs/SourceBasedCodeCoverage.html). (This document is partially based on the Clang guide.)

[#34701]: https://github.com/rust-lang/rust/issues/34701
[`llvm.instrprof.increment`]: https://llvm.org/docs/LangRef.html#llvm-instrprof-increment-intrinsic
[LLVM Code Coverage Mapping Format]: https://llvm.org/docs/CoverageMappingFormat.html
[rustc-dev-guide-how-to-build-and-run]: https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html
[`llvm-profdata merge`]: https://llvm.org/docs/CommandGuide/llvm-profdata.html#profdata-merge 
[`llvm-cov report`]: https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-report 
[`llvm-cov show`]: https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-show 
