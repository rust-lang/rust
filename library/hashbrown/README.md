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

[SwissTable]: https://abseil.io/blog/20180927-swisstables
[here]: https://github.com/abseil/abseil-cpp/blob/master/absl/container/internal/raw_hash_set.h
[CppCon talk]: https://www.youtube.com/watch?v=ncHmEUmJZf4

## [Change log](CHANGELOG.md)

## Features

- Drop-in replacement for the standard library `HashMap` and `HashSet` types.
- Uses `FxHash` as the default hasher, which is much faster than SipHash.
- Around 2x faster than `FxHashMap` and 8x faster than the standard `HashMap`.
- Lower memory usage: only 1 byte of overhead per entry instead of 8.
- Compatible with `#[no_std]` (currently requires nightly for the `alloc` crate).
- Empty hash maps do not allocate any memory.
- SIMD lookups to scan multiple hash entries in parallel.

## Performance

Compared to `std::collections::HashMap`:

```
 name               stdhash ns/iter  hashbrown ns/iter  diff ns/iter    diff %  speedup
 find_existing      23,831           2,935                   -20,896   -87.68%   x 8.12
 find_nonexisting   25,326           2,283                   -23,043   -90.99%  x 11.09
 get_remove_insert  124              25                          -99   -79.84%   x 4.96
 grow_by_insertion  197              177                         -20   -10.15%   x 1.11
 hashmap_as_queue   72               18                          -54   -75.00%   x 4.00
 new_drop           14               0                           -14  -100.00%    x inf
 new_insert_drop    78               55                          -23   -29.49%   x 1.42
```

Compared to `rustc_hash::FxHashMap` (standard `HashMap` using `FxHash` instead of `SipHash`):

```
 name               fxhash ns/iter  hashbrown ns/iter  diff ns/iter    diff %  speedup
 find_existing      5,951           2,935                    -3,016   -50.68%   x 2.03
 find_nonexisting   4,637           2,283                    -2,354   -50.77%   x 2.03
 get_remove_insert  29              25                           -4   -13.79%   x 1.16
 grow_by_insertion  160             177                          17    10.62%   x 0.90
 hashmap_as_queue   22              18                           -4   -18.18%   x 1.22
 new_drop           9               0                            -9  -100.00%    x inf
 new_insert_drop    64              55                           -9   -14.06%   x 1.16
```

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
hashbrown = "0.4"
```

This crate has the following Cargo features:

- `nightly`: Enables nightly-only features: `no_std` support and `#[may_dangle]`.
- `serde`: Enables serde serialization support.
- `rayon`: Enables rayon parallel iterator support.

## License

Licensed under either of:

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
