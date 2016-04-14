# Miri

[[slides](https://solson.me/miri-slides.pdf)]
[[report](https://solson.me/miri-report.pdf)]

An experimental interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR). This project began as part of my work for the
undergraduate research course at the [University of Saskatchewan][usask].

## Download Rust nightly

I currently recommend that you install [multirust][multirust] and then use it to
install the current rustc nightly version that works with Miri:

```sh
multirust update nightly-2016-04-11
```

## Build

```sh
multirust run nightly-2016-04-11 cargo build
```

## Run a test

```sh
multirust run nightly-2016-04-11 cargo run -- \
  --sysroot $HOME/.multirust/toolchains/nightly-2016-04-11 \
  test/filename.rs
```

If you are using [rustup][rustup] (the name of the multirust rewrite in Rust),
the `sysroot` path will also include your build target (e.g.
`$HOME/.multirust/toolchains/nightly-2016-04-11-x86_64-apple-darwin`). You can
see the current toolchain's directory by running `rustup which cargo` (ignoring
the trailing `/bin/cargo).

If you installed without using multirust or rustup, you'll need to adjust the
command to run your cargo and set the `sysroot` to the directory where your
Rust compiler is installed (`$sysroot/bin/rustc` should be a valid path).

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
[rustup]: https://www.rustup.rs