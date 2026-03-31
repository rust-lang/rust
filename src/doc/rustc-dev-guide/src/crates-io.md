# crates.io dependencies

The Rust compiler supports building with some dependencies from `crates.io`.

Rust Forge has [official policy for vetting new dependencies].

## Permitted dependencies

The `tidy` tool has [a list of crates that are allowed].
To add a dependency that is not already in the compiler, you will need to add it to the list.

[a list of crates that are allowed]: https://github.com/rust-lang/rust/blob/9d1b2106e23b1abd32fce1f17267604a5102f57a/src/tools/tidy/src/deps.rs#L73
[official policy for vetting new dependencies]: https://forge.rust-lang.org/compiler/third-party-out-of-tree#third-party-crates
