# `profile`

The tracking issue for this feature is: [#42524](https://github.com/rust-lang/rust/issues/42524).

------------------------

This feature allows the generation of code coverage reports.

Set the `-Zprofile` compiler flag in order to enable gcov profiling.

For example:
```Bash
cargo new testgcov --bin
cd testgcov
export RUSTFLAGS="-Zprofile"
cargo build
cargo run
```

Once you've built and run your program, files with the `gcno` (after build) and `gcda` (after execution) extensions will be created.
You can parse them with [llvm-cov gcov](https://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-gcov) or [grcov](https://github.com/mozilla/grcov).
