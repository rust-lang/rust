An interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR).

## Debugging

You can get detailed, statement-by-statement traces by setting the `RUST_LOG`
environment variable to `rustc_mir::miri=trace`. These traces are indented based on call stack
depth. You can get a much less verbose set of information with other logging
levels such as `warn`.

## License

Licensed under either of
  * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or
    http://opensource.org/licenses/MIT) at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.
