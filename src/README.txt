This is preliminary version of the Rust compiler.

Source layout:

comp/              The self-hosted compiler

cargo/             The package manager

libcore/           The core library (imported and linked by default)
libstd/            The standard library (slightly more peripheral code)

rustllvm/          LLVM support code

rt/                The runtime system
rt/rust_*.cpp      - The majority of the runtime services
rt/isaac           - The PRNG used for pseudo-random choices in the runtime
rt/bigint          - The bigint library used for the 'big' type
rt/uthash          - Small hashtable-and-list library for C, used in runtime
rt/libuv           - The library used for async IO in the runtime
rt/{sync,util}     - Small utility classes for the runtime.

test/              Testsuite
test/compile-fail  - Tests that should fail to compile
test/run-fail      - Tests that should compile, run and fail
test/run-pass      - Tests that should compile, run and succeed
test/bench         - Benchmarks and miscellanea

Please be gentle, it's a work in progress.
