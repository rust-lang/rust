# rust-demangler

_Demangles rustc mangled names._

`rust-demangler` supports the requirements of the [`llvm-cov show -Xdemangler`
option](https://llvm.org/docs/CommandGuide/llvm-cov.html#cmdoption-llvm-cov-show-xdemangler),
to perform Rust-specific symbol demangling:

> _The demangler is expected to read a newline-separated list of symbols from
> stdin and write a newline-separated list of the same length to stdout._

To use `rust-demangler` with `llvm-cov` for example:

```shell
$ TARGET="${PWD}/build/x86_64-unknown-linux-gnu"
$ "${TARGET}"/llvm/bin/llvm-cov show \
  --Xdemangler=path/to/rust-demangler \
  --instr-profile=main.profdata ./main --show-line-counts-or-regions
```

`rust-demangler` is a Rust "extended tool", used in Rust compiler tests, and
optionally included in Rust distributions that enable coverage profiling. Symbol
demangling is implemented using the
[rustc-demangle](https://crates.io/crates/rustc-demangle) crate.

_(Note, for Rust developers, the third-party tool
[`rustfilt`](https://crates.io/crates/rustfilt) also supports `llvm-cov` symbol
demangling. `rustfilt` is a more generalized tool that searches any body of
text, using pattern matching, to find and demangle Rust symbols.)_

## License

Rust-demangler is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](/LICENSE-APACHE) and [LICENSE-MIT](/LICENSE-MIT) for details.
