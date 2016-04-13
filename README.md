# Miri

An experimental interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR). This project began as a part of my course work for
an undergraduate research course at the [University of Saskatchewan][usask].

## Download Rust nightly

I currently recommend that you install [multirust][multirust] and then use it to
install the current rustc nightly version that works with Miri:

```sh
multirust update nightly-2016-04-05
```

## Build

```sh
multirust run nightly-2016-04-05 cargo build
```

## Run a test

```sh
multirust run nightly-2016-04-05 cargo run -- \
  --sysroot $HOME/.multirust/toolchains/nightly-2016-04-05 \
  test/filename.rs
```

If you installed without using multirust, you'll need to adjust the command to
run your cargo and set the `sysroot` to the directory where your rust compiler
is installed (`$sysroot/bin/rustc` should be a valid path).

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

[rust]: https://www.rust-lang.org/
[mir]: https://github.com/rust-lang/rfcs/blob/master/text/1211-mir.md
[usask]: https://www.usask.ca/
[multirust]: https://github.com/brson/multirust
