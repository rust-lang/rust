# num_cpus

A replacement for the deprecated `std::os::num_cpus`.

## Usage

Add to Cargo.toml:

```
[dependencies]
num_cpus = "0.2"
```

In your `main.rs` or `lib.rs`:

```rust
extern crate num_cpus;

// elsewhere
let num = num_cpus::get();
```

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
