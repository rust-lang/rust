# `linker-features`

--------------------

The `-Zlinker-features` compiler flag allows enabling or disabling specific features used during
linking, and is intended to be stabilized under the codegen options as `-Clinker-features`.

These feature flags are a flexible extension mechanism that is complementary to linker flavors,
designed to avoid the combinatorial explosion of having to create a new set of flavors for each
linker feature we'd want to use.

For example, this design allows:
- default feature sets for principal flavors, or for specific targets.
- flavor-specific features: for example, clang offers automatic cross-linking with `--target`, which
  gcc-style compilers don't support. The *flavor* is still a C/C++ compiler, and we don't want to
  multiply the number of flavors for this use-case. Instead, we can have a single `+target` feature.
- umbrella features: for example, if clang accumulates more features in the future than just the
  `+target` above. That could be modeled as `+clang`.
- niche features for resolving specific issues: for example, on Apple targets the linker flag
  implementing the `as-needed` native link modifier (#99424) is only possible on sufficiently recent
  linker versions.
- still allows for discovery and automation, for example via feature detection. This can be useful
  in exotic environments/build systems.

The flag accepts a comma-separated list of features, individually enabled (`+features`) or disabled
(`-features`), though currently only one is exposed on the CLI:
- `lld`: to toggle using the lld linker, either the system-installed binary, or the self-contained
  `rust-lld` linker.

As described above, this list is intended to grow in the future.

One of the most common uses of this flag will be to toggle self-contained linking with `rust-lld` on
and off: `-Clinker-features=+lld -Clink-self-contained=+linker` will use the toolchain's `rust-lld`
as the linker. Inversely, `-Clinker-features=-lld` would opt out of that, if the current target had
self-contained linking enabled by default.
