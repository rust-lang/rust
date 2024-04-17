# miniz_oxide

A fully safe, pure rust replacement for the [miniz](https://github.com/richgel999/miniz) DEFLATE/zlib encoder/decoder.
The main intention of this crate is to be used as a back-end for the [flate2](https://github.com/alexcrichton/flate2-rs), but it can also be used on its own. Using flate2 with the ```rust_backend``` feature provides an easy to use streaming API for miniz_oxide.

The library is fully [no_std](https://docs.rust-embedded.org/book/intro/no-std.html). By default, the `with-alloc` feature is enabled, which requires the use of the `alloc` and `collection` crates as it allocates memory.

The `std` feature additionally turns on things only available if `no_std` is not used. Currently this only means implementing [Error](https://doc.rust-lang.org/stable/std/error/trait.Error.html) for the `DecompressError` error struct returned by the simple decompression functions if enabled together with `with-alloc`.

Using the library with `default-features = false` removes the dependency on `alloc`
and `collection` crates, making it suitable for systems without an allocator.
Running without allocation reduces crate functionality:

- The `deflate` module is removed completely
- Some `inflate` functions which return a `Vec` are removed

miniz_oxide 0.5.x and 0.6.x Requires at least rust 1.40.0 0.3.x requires at least rust 0.36.0.

miniz_oxide features no use of unsafe code.

miniz_oxide can optionally be made to use a simd-accelerated version of adler32 via the [simd-adler32](https://crates.io/crates/simd-adler32) crate by enabling the 'simd' feature. This is not enabled by default as due to the use of simd intrinsics, the simd-adler32 has to use unsafe. The default setup uses the [adler](https://crates.io/crates/adler) crate which features no unsafe code.

## Usage
Simple compression/decompression:
```rust

use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::decompress_to_vec;

fn roundtrip(data: &[u8]) {
    // Compress the input
    let compressed = compress_to_vec(data, 6);
    // Decompress the compressed input and limit max output size to avoid going out of memory on large/malformed input.
    let decompressed = decompress_to_vec_with_limit(compressed.as_slice(), 60000).expect("Failed to decompress!");
    // Check roundtrip succeeded
    assert_eq!(data, decompressed);
}

fn main() {
    roundtrip("Hello, world!".as_bytes());
}

```
These simple functions will do everything in one go and are thus not recommended for use cases outside of prototyping/testing as real world data can have any size and thus result in very large memory allocations for the output Vector. Consider using miniz_oxide via [flate2](https://github.com/alexcrichton/flate2-rs) which makes it easy to do streaming (de)compression or the low-level streaming functions instead.
