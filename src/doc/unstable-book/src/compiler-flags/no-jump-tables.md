# `no-jump-tables`

The tracking issue for this feature is [#116592](https://github.com/rust-lang/rust/issues/116592)

---

This option enables the `-fno-jump-tables` flag for LLVM, which makes the
codegen backend avoid generating jump tables when lowering switches.

This option adds the LLVM `no-jump-tables=true` attribute to every function.

The option can be used to help provide protection against
jump-oriented-programming (JOP) attacks, such as with the linux kernel's [IBT].

```sh
RUSTFLAGS="-Zno-jump-tables" cargo +nightly build -Z build-std
```

[IBT]: https://www.phoronix.com/news/Linux-IBT-By-Default-Tip
