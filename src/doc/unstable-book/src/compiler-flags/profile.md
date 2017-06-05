# `profile`

The tracking issue for this feature is: None

------------------------

This feature allows the generation of code coverage reports.

Set the compiler flags `-Ccodegen-units=1 -Clink-dead-code -Cpasses=insert-gcov-profiling -Zno-landing-pads` to enable gcov profiling.

For example:
```Bash
cargo new testgcov --bin
cd testgcov
export RUSTFLAGS="-Ccodegen-units=1 -Clink-dead-code -Cpasses=insert-gcov-profiling -Zno-landing-pads"
cargo build
cargo run
```

Once you've built and run your program, files with the `gcno` (after build) and `gcda` (after execution) extensions will be created.
You can parse them with [llvm-cov gcov](http://llvm.org/docs/CommandGuide/llvm-cov.html#llvm-cov-gcov) or [grcov](https://github.com/marco-c/grcov).
