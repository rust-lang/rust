# Add support for a new function attribute

To add support for a new function attribute in libgccjit, you need to do the following steps:

 1. Copy the corresponding function from `c-family/c-attribs.cc` into `jit/dummy-frontend.cc`. For example if you add the `target` attribute, the function name will be `handle_target_attribute`.
 2. Copy the corresponding entry from the `c_common_attribute_table` variable in the `c-family/c-attribs.cc` file into the `jit_attribute_table` variable in `jit/dummy-frontend.cc`.
 3. Add a new variant in the `gcc_jit_fn_attribute` enum in the `jit/libgccjit.h` file.
 4. Add a test to ensure the attribute is correctly applied in `gcc/testsuite/jit.dg/`. Take a look at `gcc/testsuite/jit.dg/test-nonnull.c` if you want an example.
 5. Run the example like this (in your `gcc-build` folder): `make check-jit RUNTESTFLAGS="-v -v -v jit.exp=jit.dg/test-nonnull.c"`

Once done, you need to update the [gccjit.rs] crate to add the new enum variant in the corresponding enum (`FnAttribute`).

Finally, you need to update this repository by calling the relevant API you added in [gccjit.rs].

To test it, build `gcc`, run `cargo update -p gccjit` and then you can test the generated output for a given Rust crate.

[gccjit.rs]: https://github.com/rust-lang/gccjit.rs
