# crates.io Dependencies

The Rust compiler supports building with some dependencies from `crates.io`.
For example, `log` and `env_logger` come from `crates.io`.

In general, you should avoid adding dependencies to the compiler for several
reasons:

- The dependency may not be high quality or well-maintained, whereas we want
  the compiler to be high-quality.
- The dependency may not be using a compatible license.
- The dependency may have transitive dependencies that have one of the above
  problems.

As of <!-- date: 2022-02 --> February 2022, there is no official policy for vetting
new dependencies to the compiler. Generally, new dependencies are not added
to the compiler unless there is a good reason to do so.

## Permitted dependencies

The `tidy` tool has [a list of crates that are allowed]. To add a
dependency that is not already in the compiler, you will need to add it to the list.

[a list of crates that are allowed]: https://github.com/rust-lang/rust/blob/19ecce332e56941ea0dd2a805270faa102acdb14/src/tools/tidy/src/deps.rs#L59
