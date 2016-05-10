# Miri

[[slides](https://solson.me/miri-slides.pdf)]
[[report](https://solson.me/miri-report.pdf)]

An experimental interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR). This project began as part of my work for the
undergraduate research course at the [University of Saskatchewan][usask].

[![Build Status](https://travis-ci.org/tsion/miri.svg?branch=master)](https://travis-ci.org/tsion/miri)

## Download Rust nightly

I currently recommend that you install [multirust][multirust] and then use it to
install the current rustc nightly version:

```sh
multirust update nightly
```

## Build

```sh
multirust run nightly cargo build
```

If Miri fails to build, it's likely because a change in the latest nightly
compiler broke it. You could try an older nightly with `multirust update
nightly-<date>` where `<date>` is a few days or weeks ago, e.g. `2016-05-20` for
May 20th. Otherwise, you could notify me in an issue or on IRC. Or, if you know
how to fix it, you could send a PR. :smile:

## Run a test

```sh
multirust run nightly cargo run -- \
  --sysroot $HOME/.multirust/toolchains/nightly \
  test/filename.rs
```

If you are using [rustup][rustup] (the name of the multirust rewrite in Rust),
the `sysroot` path will also include your build target (e.g.
`$HOME/.multirust/toolchains/nightly-x86_64-apple-darwin`). You can see the
current toolchain's directory by running `rustup which cargo` (ignoring the
trailing `/bin/cargo`).

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
