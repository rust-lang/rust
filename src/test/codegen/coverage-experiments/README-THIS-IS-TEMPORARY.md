# codegen/coverage-experiments
*<h2>THIS DIRECTORY IS TEMPORARY</h2>*

This directory contains some work-in-progress (WIP) code used for experimental development and
testing of Rust Coverage feature development.

The code in this directory will be removed, or migrated into product tests, when the Rust
Coverage feature is complete.

[TOC]

## Development Notes

### config.toml

config.toml probably requires (I should verify that intrinsic `llvm.instrprof.increment`
code generation ONLY works with this config option):

  profiler = true

## First build

```shell
./x.py clean
./x.py build -i --stage 1 src/libstd
```

## Incremental builds *IF POSSIBLE!*

```shell
./x.py build -i --stage 1 src/libstd --keep-stage 1
```

*Note: Some changes made for Rust Coverage required the full build (without `--keep-stage 1`), and in some cases, required `./x.py clean` first!. Occassionally I would get errors when building or when compiling a test program with `--Zinstrument-coverage` that work correctly only after a full clean and build.*

## Compile a test program with LLVM coverage instrumentation

*Note: This PR is still a work in progress. At the time of this writing, the `llvm.instrprof.increment` intrinsic is injected, and recognized by the LLVM code generation stage, but it does not appear to be included in the final binary. This is not surprising since other steps are still to be implemented, such as generating the coverage map. See the suggested additional `llvm` flags for ways to verify the `llvm` passes at least get the right intrinsic.*

Suggested debug configuration to confirm Rust coverage features:
```shell
$ export RUSTC_LOG=rustc_codegen_llvm::intrinsic,rustc_mir::transform::instrument_coverage=debug
```

Ensure the new compiled `rustc` is used (the path below, relative to the `rust` code repository root, is an example only):

```shell
$ build/x86_64-unknown-linux-gnu/stage1/bin/rustc \
  src/test/codegen/coverage-experiments/just_main.rs \
  -Zinstrument-coverage
```

### About the test programs in coverage-experiments/src/

The `coverage-experiments/src/` directory contains some sample (and very simple) Rust programs used to analyze Rust compiler output at various stages, with or without the Rust code coverage compiler option. For now, these are only used for the in-progress development and will be removed at a future date. (These are *not* formal test programs.)

The src director may also contain some snapshots of mir output from experimentation, particularly if the saved snapshots highlight results that are important to the future development, individually or when compared with other output files.

Be aware that some of the files and/or comments may be outdated.

### Additional `llvm` flags (append to the `rustc` command)

These optional flags generate additional files and/or terminal output. LLVM's `-print-before=all` should show the `instrprof.increment` intrinsic with arguments computed by the experimental Rust coverage feature code:

```shell
  --emit llvm-ir \
  -Zverify-llvm-ir \
  -Zprint-llvm-passes \
  -Csave-temps \
  -Cllvm-args=-print-before-all
```

### Additional flags for MIR analysis and transforms

These optional flags generate a directory with many files representing the MIR as text (`.mir` files) and as a visual graph (`.dot` files) rendered by `graphviz`. (**Some IDEs, such as `VSCode` have `graphviz` extensions.**)

```shell
  -Zdump-mir=main \
  -Zdump-mir-graphviz
```

### Flags I've used but appear to be irrelvant to `-Zinstrument-coverage` after all:
```shell
  # -Zprofile
  # -Ccodegen-units=1
  # -Cinline-threshold=0
  # -Clink-dead-code
  # -Coverflow-checks=off
```

## Run the test program compiled with code coverage instrumentation (maybe):

As stated above, at the time of this writing, this work-in-progress seems to generate `llvm.instrprof.increment` intrinsic calls correctly, and are visibile in early `llvm` code generation passes, but are eventually stripped.

The test program should run as expected, currently does not generate any coverage output.

*Example:*

```shell
  $ src/test/codegen/coverage-experiments/just_main
  hello world! (should be covered)
```

### Running the coverage-enabled `rustc` compiler in the `lldb` debugger:

For example, to verify the intrinsic is codegen'ed, set breakpoint in `lldb` where it validates a certain instruction is the `llvm.instrprof.increment` instruction.

First, update config.toml for debugging:

```toml
  [llvm]
  optimize = false
  release-debuginfo = true

  [rust]
  debug = true
  debuginfo-level = 2
```

*(Note, in case this is relevant after all, I also have the following changes; but I don't think I need them:)*

```toml
  # Add and uncomment these if relevant/useful:
  # codegen-units = 0
  # python = '/usr/bin/python3.6'
```

Run the compiler with additional flags as needed:

```shell
lldb \
  build/x86_64-unknown-linux-gnu/stage1/bin/rustc \
  -- \
  src/test/codegen/coverage-experiments/just_main.rs \
  -Zinstrument-coverage \
  -Zdump-mir=main \
  -Zdump-mir-graphviz
```

Note the specific line numbers may be different:

```c++
(lldb) b lib/Transforms/Instrumentation/InstrProfiling.cpp:418
(lldb) r

Process 93855 stopped
* thread #6, name = 'rustc', stop reason = breakpoint 2.1
    frame #0: 0x00007fffedff7738 librustc_driver-5a0990d8d18fb2b4.so`llvm::InstrProfiling::lowerIntrinsics(this=0x00007fffcc001d40, F=0x00007fffe4552198) at InstrProfiling.cpp:418:23
   415        auto Instr = I++;
   416        InstrProfIncrementInst *Inc = castToIncrementInst(&*Instr);
   417        if (Inc) {
-> 418          lowerIncrement(Inc);
   419          MadeChange = true;
   420        } else if (auto *Ind = dyn_cast<InstrProfValueProfileInst>(Instr)) {
   421          lowerValueProfileInst(Ind);
(lldb)
```