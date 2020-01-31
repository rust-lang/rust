hashbrown
=========

[![Build Status](https://travis-ci.com/rust-lang/hashbrown.svg?branch=master)](https://travis-ci.com/rust-lang/hashbrown)
[![Crates.io](https://img.shields.io/crates/v/hashbrown.svg)](https://crates.io/crates/hashbrown)
[![Documentation](https://docs.rs/hashbrown/badge.svg)](https://docs.rs/hashbrown)

This crate is a Rust port of Google's high-performance [SwissTable] hash
map, adapted to make it a drop-in replacement for Rust's standard `HashMap`
and `HashSet` types.

The original C++ version of SwissTable can be found [here], and this
[CppCon talk] gives an overview of how the algorithm works.

Since Rust 1.36, this is now the `HashMap` implementation for the Rust standard
library. However you may still want to use this crate instead since it works
in environments without `std`, such as embedded systems and kernels.

[SwissTable]: https://abseil.io/blog/20180927-swisstables
[here]: https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h
[CppCon talk]: https://www.youtube.com/watch?v=ncHmEUmJZf4

## [Change log](CHANGELOG.md)

## Features

- Drop-in replacement for the standard library `HashMap` and `HashSet` types.
- Uses `AHash` as the default hasher, which is much faster than SipHash.
- Around 2x faster than the previous standard library `HashMap`.
- Lower memory usage: only 1 byte of overhead per entry instead of 8.
- Compatible with `#[no_std]` (but requires a global allocator with the `alloc` crate).
- Empty hash maps do not allocate any memory.
- SIMD lookups to scan multiple hash entries in parallel.

## Performance

Compared to the previous implementation of `std::collections::HashMap` (Rust 1.35).

With the hashbrown default AHash hasher (not HashDoS-resistant):

```text
 name                       oldstdhash ns/iter  hashbrown ns/iter  diff ns/iter   diff %  speedup 
 insert_ahash_highbits        20,846              7,397                   -13,449  -64.52%   x 2.82 
 insert_ahash_random          20,515              7,796                   -12,719  -62.00%   x 2.63 
 insert_ahash_serial          21,668              7,264                   -14,404  -66.48%   x 2.98 
 insert_erase_ahash_highbits  29,570              17,498                  -12,072  -40.83%   x 1.69 
 insert_erase_ahash_random    39,569              17,474                  -22,095  -55.84%   x 2.26 
 insert_erase_ahash_serial    32,073              17,332                  -14,741  -45.96%   x 1.85 
 iter_ahash_highbits          1,572               2,087                       515   32.76%   x 0.75 
 iter_ahash_random            1,609               2,074                       465   28.90%   x 0.78 
 iter_ahash_serial            2,293               2,120                      -173   -7.54%   x 1.08 
 lookup_ahash_highbits        3,460               4,403                       943   27.25%   x 0.79 
 lookup_ahash_random          6,377               3,911                    -2,466  -38.67%   x 1.63 
 lookup_ahash_serial          3,629               3,586                       -43   -1.18%   x 1.01 
 lookup_fail_ahash_highbits   5,286               3,411                    -1,875  -35.47%   x 1.55 
 lookup_fail_ahash_random     12,365              4,171                    -8,194  -66.27%   x 2.96 
 lookup_fail_ahash_serial     4,902               3,240                    -1,662  -33.90%   x 1.51 
```

With the libstd default SipHash hasher (HashDoS-resistant):

```text
 name                       oldstdhash ns/iter  hashbrown ns/iter  diff ns/iter   diff %  speedup 
 insert_std_highbits        32,598              20,199                  -12,399  -38.04%   x 1.61 
 insert_std_random          29,824              20,760                   -9,064  -30.39%   x 1.44 
 insert_std_serial          33,151              17,256                  -15,895  -47.95%   x 1.92 
 insert_erase_std_highbits  74,731              48,735                  -25,996  -34.79%   x 1.53 
 insert_erase_std_random    73,828              47,649                  -26,179  -35.46%   x 1.55 
 insert_erase_std_serial    73,864              40,147                  -33,717  -45.65%   x 1.84 
 iter_std_highbits          1,518               2,264                       746   49.14%   x 0.67 
 iter_std_random            1,502               2,414                       912   60.72%   x 0.62 
 iter_std_serial            6,361               2,118                    -4,243  -66.70%   x 3.00 
 lookup_std_highbits        21,705              16,962                   -4,743  -21.85%   x 1.28 
 lookup_std_random          21,654              17,158                   -4,496  -20.76%   x 1.26 
 lookup_std_serial          18,726              14,509                   -4,217  -22.52%   x 1.29 
 lookup_fail_std_highbits   25,852              17,323                   -8,529  -32.99%   x 1.49 
 lookup_fail_std_random     25,913              17,760                   -8,153  -31.46%   x 1.46 
 lookup_fail_std_serial     22,648              14,839                   -7,809  -34.48%   x 1.53 
```

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
hashbrown = "0.7"
```

Then:

```rust
use hashbrown::HashMap;

let mut map = HashMap::new();
map.insert(1, "one");
```

This crate has the following Cargo features:

- `nightly`: Enables nightly-only features: `#[may_dangle]`.
- `serde`: Enables serde serialization support.
- `rayon`: Enables rayon parallel iterator support.
- `raw`: Enables access to the experimental and unsafe `RawTable` API.
- `inline-more`: Adds inline hints to most functions, improving run-time performance at the cost
  of compilation time. (enabled by default)
- `ahash`: Compiles with ahash as default hasher. (enabled by default)
- `ahash-compile-time-rng`: Activates the `compile-time-rng` feature of ahash, to increase the
   DOS-resistance, but can result in issues for `no_std` builds. More details in
   [issue#124](https://github.com/rust-lang/hashbrown/issues/124). (enabled by default)

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
