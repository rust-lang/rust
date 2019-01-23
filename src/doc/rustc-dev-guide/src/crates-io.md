# crates.io Dependencies

The rust compiler supports building with some dependencies from `crates.io`.
For example, `log` and `env_logger` come from `crates.io`.

In general, you should avoid adding dependencies to the compiler for several
reasons:

- The dependency may not be high quality or well-maintained, whereas we want
  the compiler to be high-quality.
- The dependency may not be using a compatible license.
- The dependency may have transitive dependencies that have one of the above
  problems.

TODO: what is the vetting process?

## Whitelist

The `tidy` tool has a [whitelist] of crates that are allowed. To add a
dependency that is not already in the compiler, you will need to add it to this
whitelist.

[whitelist]: https://github.com/rust-lang/rust/blob/659994627234ce7d95a1a52ad8756ce661059adf/src/tools/tidy/src/deps.rs#L56
