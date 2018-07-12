# `test-generator`

This is a tool to generate test cases for the `libm` crate.

The generator randomly creates inputs for each math function, then proceeds to compute the
expected output for the given function by running the MUSL *C implementation* of the function and
finally it packs the test cases as a Cargo test file. For this reason, this generator **must**
always be compiled for the `x86_64-unknown-linux-musl` target.
