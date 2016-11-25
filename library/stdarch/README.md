stdsimd
=======
Experiments for adding SIMD support to Rust's standard library.

This is a **work in progress**.

### Approach

The main goal is to expose APIs defined by *vendors* with the least amount of
abstraction possible. On x86, for example, the API should correspond to that
provided by `emmintrin.h`.
