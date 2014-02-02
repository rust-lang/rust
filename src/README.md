This is a preliminary version of the Rust compiler, libraries and tools.

Source layout:

| Path                | Description                                               |
| ------------------- | --------------------------------------------------------- |
| `librustc/`         | The self-hosted compiler                                  |
| `libstd/`           | The standard library (imported and linked by default)     |
| `libextra/`         | The "extras" library (slightly more peripheral code)      |
| `libgreen/`         | The M:N runtime library                                   |
| `libnative/`        | The 1:1 runtime library                                   |
| `libsyntax/`        | The Rust parser and pretty-printer                        |
| ------------------- | --------------------------------------------------------- |
| `rt/`               | The runtime system                                        |
| `rt/rust_*.c`       | - Some of the runtime services                            |
| `rt/vg`             | - Valgrind headers                                        |
| `rt/msvc`           | - MSVC support                                            |
| `rt/sundown`        | - The Markdown library used by rustdoc                    |
| ------------------- | --------------------------------------------------------- |
| `compiletest/`      | The test runner                                           |
| `test/`             | Testsuite                                                 |
| `test/codegen`      | - Tests for the LLVM IR infrastructure                    |
| `test/compile-fail` | - Tests that should fail to compile                       |
| `test/debug-info`   | - Tests for the `debuginfo` tool                          |
| `test/run-fail`     | - Tests that should compile, run and fail                 |
| `test/run-make`     | - Tests that depend on a Makefile infrastructure          |
| `test/run-pass`     | - Tests that should compile, run and succeed              |
| `test/bench`        | - Benchmarks and miscellaneous                            |
| `test/pretty`       | - Pretty-printer tests                                    |
| `test/auxiliary`    | - Dependencies of tests                                   |
| ------------------- | --------------------------------------------------------- |
| `librustdoc/`       | The Rust API documentation tool                           |
| `libuv/`            | The libuv submodule                                       |
| ------------------- | --------------------------------------------------------- |
| `llvm/`             | The LLVM submodule                                        |
| `rustllvm/`         | LLVM support code                                         |
| ------------------- | --------------------------------------------------------- |
| `etc/`              | Scripts, editors support, misc                            |
