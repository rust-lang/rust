# `source-based-code-coverage`

The tracking issue for this feature is: [#79121].

------------------------

## Introduction

The Rust compiler includes two code coverage implementations:

* A GCC-compatible, gcov-based coverage implementation, enabled with [`-Zprofile`], which operates on DebugInfo.
* A source-based code coverage implementation, enabled with `-Zinstrument-coverage`, which uses LLVM's native coverage instrumentation to generate very precise coverage data.

This document describes how to enable and use the LLVM instrumentation-based coverage, via the `-Zinstrument-coverage` compiler flag.

## How it works

When `-Zinstrument-coverage` is enabled, the Rust compiler enhances rust-based libraries and binaries by:

* Automatically injecting calls to an LLVM intrinsic ([`llvm.instrprof.increment`]), at functions and branches in compiled code, to increment counters when conditional sections of code are executed.
* Embedding additional information in the data section of each library and binary (using the [LLVM Code Coverage Mapping Format] _Version 4_, supported _only_ in LLVM 11 and up), to define the code regions (start and end positions in the source code) being counted.

When running a coverage-instrumented program, the counter values are written to a `profraw` file at program termination. LLVM bundles tools that read the counter results, combine those results with the coverage map (embedded in the program binary), and generate coverage reports in multiple formats.

## Enable coverage profiling in the Rust compiler

Rust's source-based code coverage requires the Rust "profiler runtime". Without it, compiling with `-Zinstrument-coverage` generates an error that the profiler runtime is missing.

The Rust `nightly` distribution channel should include the profiler runtime, by default.

*IMPORTANT:* If you are building the Rust compiler from the source distribution, the profiler runtime is *not* enabled in the default `config.toml.example`. Edit your `config.toml` file and ensure the `profiler` feature is set it to `true`:

```toml
# Build the profiler runtime (required when compiling with options that depend
# on this runtime, such as `-C profile-generate` or `-Z instrument-coverage`).
profiler = true
```

If changed, rebuild the Rust compiler (see [rustc-dev-guide-how-to-build-and-run]).

### Building the demangler

LLVM coverage reporting tools generate results that can include function names and other symbol references, and the raw coverage results report symbols using the compiler's "mangled" version of the symbol names, which can be difficult to interpret. To work around this issue, LLVM coverage tools also support a user-specified symbol name demangler.

One option for a Rust demangler is [`rustfilt`], which can be installed with:

```shell
cargo install rustfilt
```

Another option, if you are building from the Rust compiler source distribution, is to use the `rust-demangler` tool included in the Rust source distribution, which can be built with:

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

If `cargo` is not configured to use your `profiler`-enabled version of `rustc`, set the path explicitly via the `RUSTC` environment variable. Here is another example, using a `stage1` build of `rustc` to compile an `example` binary (from the [`json5format`] crate):

```shell
$ RUSTC=$HOME/rust/build/x86_64-unknown-linux-gnu/stage1/bin/rustc \
    RUSTFLAGS="-Zinstrument-coverage" \
    cargo build --example formatjson5
```

Note that some compiler options, combined with `-Zinstrument-coverage`, can produce LLVM IR and/or linked binaries that are incompatible with LLVM coverage maps. For example, coverage requires references to actual functions in LLVM IR. If any covered function is optimized out, the coverage tools may not be able to process the coverage results. If you need to pass additional options, with coverage enabled, test them early, to confirm you will get the coverage results you expect.

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

## Installing LLVM coverage tools

LLVM's supplies two tools—`llvm-profdata` and `llvm-cov`—that process coverage data and generate reports. There are several ways to find and/or install these tools, but note that the coverage mapping data generated by the Rust compiler requires LLVM version 11 or higher. (`llvm-cov --version` typically shows the tool's LLVM version number.):

* The LLVM tools may be installed (or installable) directly to your OS (such as via `apt-get`, for Linux).
* If you are building the Rust compiler from source, you can optionally use the bundled LLVM tools, built from source. Those tool binaries can typically be found in your build platform directory at something like: `rust/build/x86_64-unknown-linux-gnu/llvm/bin/llvm-*`.
* You can install compatible versions of these tools via `rustup`.

The `rustup` option is guaranteed to install a compatible version of the LLVM tools, but they can be hard to find. We recommend [`cargo-bintools`], which installs Rust-specific wrappers around these and other LLVM tools, so you can invoke them via `cargo` commands!

```shell
$ rustup component add llvm-tools-preview
$ cargo install cargo-binutils
$ cargo profdata -- --help  # note the additional "--" preceeding the tool-specific arguments
```

## Creating coverage reports

Raw profiles have to be indexed before they can be used to generate coverage reports. This is done using [`llvm-profdata merge`] (or `cargo cov -- merge`), which can combine multiple raw profiles and index them at the same time:

```shell
$ llvm-profdata merge -sparse formatjson5.profraw -o formatjson5.profdata
```

Finally, the `.profdata` file is used, in combination with the coverage map (from the program binary) to generate coverage reports using [`llvm-cov report`] (or `cargo cov -- report`), for a coverage summaries; and [`llvm-cov show`] (or `cargo cov -- show`), to see detailed coverage of lines and regions (character ranges) overlaid on the original source code.

These commands have several display and filtering options. For example:

```shell
$ llvm-cov show -Xdemangler=rustfilt target/debug/examples/formatjson5 \
    -instr-profile=formatjson5.profdata \
    -show-line-counts-or-regions \
    -show-instantiations \
    -name=add_quoted_string
```

<img alt="Screenshot of sample `llvm-cov show` result, for function add_quoted_string" src="img/llvm-cov-show-01.png" class="center"/>
<br/>
<br/>

Some of the more notable options in this example include:

* `--Xdemangler=rustfilt` - the command name or path used to demangle Rust symbols (`rustfilt` in the example, but this could also be a path to the `rust-demangler` tool)
* `target/debug/examples/formatjson5` - the instrumented binary (from which to extract the coverage map)
* `--instr-profile=<path-to-file>.profdata` - the location of the `.profdata` file created by `llvm-profdata merge` (from the `.profraw` file generated by the instrumented binary)
* `--name=<exact-function-name>` - to show coverage for a specific function (or, consider using another filter option, such as `--name-regex=<pattern>`)

## Interpreting reports

There are four statistics tracked in a coverage summary:

* Function coverage is the percentage of functions that have been executed at least once. A function is considered to be executed if any of its instantiations are executed.
* Instantiation coverage is the percentage of function instantiations that have been executed at least once. Generic functions and functions generated from macros are two kinds of functions that may have multiple instantiations.
* Line coverage is the percentage of code lines that have been executed at least once. Only executable lines within function bodies are considered to be code lines.
* Region coverage is the percentage of code regions that have been executed at least once. A code region may span multiple lines: for example, in a large function body with no control flow. In other cases, a single line can contain multiple code regions: `return x || (y && z)` has countable code regions for `x` (which may resolve the expression, if `x` is `true`), `|| (y && z)` (executed only if `x` was `false`), and `return` (executed in either situation).

Of these four statistics, function coverage is usually the least granular while region coverage is the most granular. The project-wide totals for each statistic are listed in the summary.

## Test coverage

A typical use case for coverage analysis is test coverage. Rust's source-based coverage tools can both measure your tests' code coverage as percentage, and pinpoint functions and branches not tested.

The following example (using the [`json5format`] crate, for demonstration purposes) show how to generate and analyze coverage results for all tests in a crate.

Since `cargo test` both builds and runs the tests, we set both the additional `RUSTFLAGS`, to add the `-Zinstrument-coverage` flag, and `LLVM_PROFILE_FILE`, to set a custom filename for the raw profiling data generated during the test runs. Since there may be more than one test binary, apply `%m` in the filename pattern. This generates unique names for each test binary. (Otherwise, each executed test binary would overwrite the coverage results from the previous binary.)

```shell
$ RUSTFLAGS="-Zinstrument-coverage" \
    LLVM_PROFILE_FILE="json5format-%m.profraw" \
    cargo test --tests
```

Make note of the test binary file paths, displayed after the word "`Running`" in the test output:

```text
   ...
   Compiling json5format v0.1.3 ($HOME/json5format)
    Finished test [unoptimized + debuginfo] target(s) in 14.60s

     Running target/debug/deps/json5format-fececd4653271682
running 25 tests
...
test result: ok. 25 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

     Running target/debug/deps/lib-30768f9c53506dc5
running 31 tests
...
test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

You should have one ore more `.profraw` files now, one for each test binary. Run the `profdata` tool to merge them:

```shell
$ cargo profdata -- merge \
    -sparse json5format-*.profraw -o json5format.profdata
```

Then run the `cov` tool, with the `profdata` file and all test binaries:

```shell
$ cargo cov -- report \
    --use-color --ignore-filename-regex='/.cargo/registry' \
    --instr-profile=json5format.profdata \
    --object target/debug/deps/lib-30768f9c53506dc5 \
    --object target/debug/deps/json5format-fececd4653271682
$ cargo cov -- show \
    --use-color --ignore-filename-regex='/.cargo/registry' \
    --instr-profile=json5format.profdata \
    --object target/debug/deps/lib-30768f9c53506dc5 \
    --object target/debug/deps/json5format-fececd4653271682 \
    --show-instantiations --show-line-counts-or-regions \
    --Xdemangler=rustfilt | less -R
```

_Note the command line option `--ignore-filename-regex=/.cargo/registry`, which excludes the sources for dependencies from the coverage results._

### Tips for listing the binaries automatically

For `bash` users, one suggested way to automatically complete the `cov` command with the list of binaries is with a command like:

```bash
$ cargo cov -- report \
    $( \
      for file in \
        $( \
          RUSTFLAGS="-Zinstrument-coverage" \
            cargo test --tests --no-run --message-format=json \
              | jq -r "select(.profile.test == true) | .filenames[]" \
              | grep -v dSYM - \
        ); \
      do \
        printf "%s %s " -object $file; \
      done \
    ) \
  --instr-profile=json5format.profdata --summary-only # and/or other options
```

Adding `--no-run --message-format=json` to the _same_ `cargo test` command used to run
the tests (including the same environment variables and flags) generates output in a JSON
format that `jq` can easily query.

The `printf` command takes this list and generates the `--object <binary>` arguments
for each listed test binary.

### Including doc tests

The previous examples run `cargo test` with `--tests`, which excludes doc tests.[^79417]

To include doc tests in the coverage results, drop the `--tests` flag, and apply the
`-Zinstrument-coverage` flag, and some doc-test-specific options in the
`RUSTDOCFLAGS` environment variable. (The `cargo profdata` command does not change.)

```bash
$ RUSTFLAGS="-Zinstrument-coverage" \
  RUSTDOCFLAGS="-Zinstrument-coverage -Zunstable-options --persist-doctests target/debug/doctestbins" \
  LLVM_PROFILE_FILE="json5format-%m.profraw" \
    cargo test
$ cargo profdata -- merge \
    -sparse json5format-*.profraw -o json5format.profdata
```

The `-Zunstable-options --persist-doctests` flag is required, to save the test binaries
(with their coverage maps) for `llvm-cov`.

```bash
$ cargo cov -- report \
    $( \
      for file in \
        $( \
          RUSTFLAGS="-Zinstrument-coverage" \
          RUSTDOCFLAGS="-Zinstrument-coverage -Zunstable-options --persist-doctests target/debug/doctestbins" \
            cargo test --no-run --message-format=json \
              | jq -r "select(.profile.test == true) | .filenames[]" \
              | grep -v dSYM - \
        ) \
        target/debug/doctestbins/*/rust_out; \
      do \
        [[ -x $file ]] && printf "%s %s " -object $file; \
      done \
    ) \
  --instr-profile=json5format.profdata --summary-only # and/or other options
```

Note, the differences in this `cargo cov` command, compared with the version without
doc tests, include:

* The `cargo test ... --no-run` command is updated with the same environment variables
  and flags used to _build_ the tests, _including_ the doc tests. (`LLVM_PROFILE_FILE`
  is only used when _running_ the tests.)
* The file glob pattern `target/debug/doctestbins/*/rust_out` adds the `rust_out`
  binaries generated for doc tests (note, however, that some `rust_out` files may not
  be executable binaries).
* `[[ -x $file ]] &&` filters the files passed on to the `printf`, to include only
  executable binaries.

[^79417]: There is ongoing work to resolve a known issue
[(#79417)](https://github.com/rust-lang/rust/issues/79417) that doc test coverage
generates incorrect source line numbers in `llvm-cov show` results.

## Other references

Rust's implementation and workflow for source-based code coverage is based on the same library and tools used to implement [source-based code coverage in Clang]. (This document is partially based on the Clang guide.)

[#79121]: https://github.com/rust-lang/rust/issues/79121
[`-Zprofile`]: profile.md
[`llvm.instrprof.increment`]: https://llvm.org/docs/LangRef.html#llvm-instrprof-increment-intrinsic
[LLVM Code Coverage Mapping Format]: https://llvm.org/docs/CoverageMappingFormat.html
[rustc-dev-guide-how-to-build-and-run]: https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html
[`rustfilt`]: https://crates.io/crates/rustfilt
[`json5format`]: https://crates.io/crates/json5format
[`cargo-bintools`]: https://crates.io/crates/cargo-bintools
[`llvm-profdata merge`]: https://llvm.org/docs/CommandGuide/llvm-profdata.html#profdata-merge
[`llvm-cov report`]: https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-report
[`llvm-cov show`]: https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-show
[source-based code coverage in Clang]: https://clang.llvm.org/docs/SourceBasedCodeCoverage.html