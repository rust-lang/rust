# `fat-lto-partitions`

------------------------

`-Zfat-lto-partitions=N` splits the merged fat-LTO module into `N` function
partitions and a trailing data partition after the LTO optimization pipeline
has run. Rustc then codegens them in parallel within the jobserver's budget.
`N` must be between 1 and 256.

The default (`1`) keeps single-module codegen. The flag applies to the LLVM
backend with `-Clto=fat`, and to a merged `-Zfat-lto-crates` core under
`-Clto=thin`.

Partitioning happens after all interprocedural optimization, so each function
is optimized exactly as without partitioning. Internal symbols referenced
across partitions become hidden symbols in the object files. LLVM groups that
must remain in one object, such as COMDAT members and ifunc resolvers, stay
together. A module containing module-level inline assembly remains intact in
the first partition because textual symbol references cannot be analyzed.
Output is deterministic for a fixed `N`; different values of `N` lay the
binary out differently.

Example:

```console
RUSTFLAGS="-Zfat-lto-partitions=16" cargo +nightly build --release
```

(with `lto = "fat"` in the Cargo profile.)
