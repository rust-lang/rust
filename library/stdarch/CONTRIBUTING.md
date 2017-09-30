# Contributing to stdsimd

The `stdsimd` crate is more than willing to accept contributions! First you'll
probably want to check out the repository and make sure that tests pass for you:

```
$ git clone https://github.com/rust-lang-nursery/stdsimd
$ cd stdsimd
$ cargo +nightly test
```

To run codegen tests, run in release mode:

```
$ cargo +nightly test --release
```

Remember that this repository requires the nightly channel of Rust! If any of
the above steps don't work, [please let us know][new]!

Next up you can [find an issue][issues] to help out on, we've selected a few
with the [`help wanted`][help] and [`impl-period`][impl] tags which could
particularly use some help. You may be most interested in [#40][vendor],
implementing all vendor intrinsics on x86. That issue's got some good pointers
about where to get started!

[new]: https://github.com/rust-lang-nursery/stdsimd/issues/new
[issues]: https://github.com/rust-lang-nursery/stdsimd/issues
[help]: https://github.com/rust-lang-nursery/stdsimd/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22
[impl]: https://github.com/rust-lang-nursery/stdsimd/issues?q=is%3Aissue+is%3Aopen+label%3Aimpl-period
[vendor]: https://github.com/rust-lang-nursery/stdsimd/issues/40

If you've got general questions feel free to [join us on gitter][gitter] and ask
around! Feel free to ping either @BurntSushi or @alexcrichton with questions.

[gitter]: https://gitter.im/rust-impl-period/WG-libs-simd
