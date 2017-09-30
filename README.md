# Miri [[slides](https://solson.me/miri-slides.pdf)] [[report](https://solson.me/miri-report.pdf)] [![Build Status](https://travis-ci.org/solson/miri.svg?branch=master)](https://travis-ci.org/solson/miri) [![Windows build status](https://ci.appveyor.com/api/projects/status/github/solson/miri?svg=true)](https://ci.appveyor.com/project/solson63299/miri)


An experimental interpreter for [Rust][rust]'s [mid-level intermediate
representation][mir] (MIR). This project began as part of my work for the
undergraduate research course at the [University of Saskatchewan][usask].

## Installing Rust

I recommend that you install [rustup][rustup] and then use it to install the
current Rust nightly version:

```sh
rustup update nightly
```

You should also make `nightly` the default version for your Miri directory by
running the following command while you're in it. If you don't do this, you can
run the later `cargo` commands by using `cargo +nightly` instead.

```sh
rustup override add nightly
```

## Building Miri

```sh
cargo build
```

If Miri fails to build, it's likely because a change in the latest nightly
compiler broke it. You could try an older nightly with `rustup update
nightly-<date>` where `<date>` is a few days or weeks ago, e.g. `2016-05-20` for
May 20th. Otherwise, you could notify me in an issue or on IRC. Or, if you know
how to fix it, you could send a PR. :smile:

## Running tests

```sh
cargo run --bin miri tests/run-pass-fullmir/vecs.rs # Or whatever test you like.
```

## Debugging

You can get detailed, statement-by-statement traces by setting the `MIRI_LOG`
environment variable to `trace`. These traces are indented based on call stack
depth. You can get a much less verbose set of information with other logging
levels such as `warn`.

## Running miri on your own project('s test suite)

Install miri as a cargo subcommand with `cargo install --debug`.
Then, inside your own project, use `cargo +nightly miri` to run your project, if it is
a bin project, or run `cargo +nightly miri test` to run all tests in your project
through miri.

## Running miri with full libstd

Per default libstd does not contain the MIR of non-polymorphic functions.  When
miri hits a call to such a function, execution terminates.  To fix this, it is
possible to compile libstd with full MIR:

```sh
rustup component add rust-src
cargo install xargo
cd xargo/
RUSTFLAGS='-Zalways-encode-mir' xargo build
```

Now you can run miri against the libstd compiled by xargo:

```sh
MIRI_SYSROOT=~/.xargo/HOST cargo run --bin miri tests/run-pass-fullmir/vecs.rs
```

Notice that you will have to re-run the last step of the preparations above when
your toolchain changes (e.g., when you update the nightly).

## Contributing and getting help

Check out the issues on this GitHub repository for some ideas. There's lots that
needs to be done that I haven't documented in the issues yet, however. For more
ideas or help with running or hacking on Miri, you can contact me (`scott`) on
Mozilla IRC in any of the Rust IRC channels (`#rust`, `#rust-offtopic`, etc).

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
[rustup]: https://www.rustup.rs
