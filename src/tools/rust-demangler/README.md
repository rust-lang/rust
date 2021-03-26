# rust-demangler

Demangles rustc mangled names.

This tool uses the [rustc-demangle](https://crates.io/crates/rustc-demangle)
crate to convert an input buffer of newline-separated mangled names into their
demangled translations.

This tool takes a list of mangled names (one per line) on standard input, and
prints a corresponding list of demangled names. The tool is designed to support
programs that can leverage a third-party demangler, such as `llvm-cov`, via the
`-Xdemangler=<path-to-demangler>` option.

To use `rust-demangler` with `llvm-cov` for example, add the `-Xdemangler=...`
option:

```shell
$ TARGET="${PWD}/build/x86_64-unknown-linux-gnu"
$ "${TARGET}"/llvm/bin/llvm-cov show \
  --Xdemangler=path/to/rust-demangler \
  --instr-profile=main.profdata ./main --show-line-counts-or-regions
```

## License

Rust-demangler is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [LICENSE-APACHE](/LICENSE-APACHE) and [LICENSE-MIT](/LICENSE-MIT) for details.
