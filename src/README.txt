This is a preliminary version of the Rust compiler, libraries and tools

Source layout:

librustc/          The self-hosted compiler

libstd/            The standard library (imported and linked by default)
libextra/          The "extras" library (slightly more peripheral code)
libsyntax/         The Rust parser and pretty-printer

rt/                The runtime system
rt/rust_*.cpp      - The majority of the runtime services
rt/isaac           - The PRNG used for pseudo-random choices in the runtime
rt/bigint          - The bigint library used for the 'big' type
rt/uthash          - Small hashtable-and-list library for C, used in runtime
rt/sync            - Concurrency utils
rt/util            - Small utility classes for the runtime.
rt/vg              - Valgrind headers
rt/msvc            - MSVC support
rt/linenoise       - a readline-like line editing library

test/              Testsuite
test/compile-fail  - Tests that should fail to compile
test/run-fail      - Tests that should compile, run and fail
test/run-pass      - Tests that should compile, run and succeed
test/bench         - Benchmarks and miscellanea
test/pretty        - Pretty-printer tests
test/auxiliary     - Dependencies of tests

compiletest/       The test runner

librustpkg/        The package manager and build system

librusti/          The JIT REPL

librustdoc/        The Rust API documentation tool

llvm/              The LLVM submodule

libuv/             The libuv submodule

rustllvm/          LLVM support code

libfuzzer/         A collection of fuzz testers

etc/               Scripts, editor support, misc
