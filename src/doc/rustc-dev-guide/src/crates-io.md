# crates.io dependencies

The Rust compiler supports building with some dependencies from `crates.io`.
Examples are `log` and `env_logger`.

In general,
you should avoid adding dependencies to the compiler for several reasons:

- The dependency may not be of high quality or well-maintained.
- The dependency may not be using a compatible license.
- The dependency may have transitive dependencies that have one of the above
  problems.

<!-- date-check: Aug 2025 -->
Note that there is no official policy for vetting new dependencies to the compiler.
Decisions are made on a case-by-case basis, during code review.

## Permitted dependencies

The `tidy` tool has [a list of crates that are allowed]. To add a
dependency that is not already in the compiler, you will need to add it to the list.

[a list of crates that are allowed]: https://github.com/rust-lang/rust/blob/9d1b2106e23b1abd32fce1f17267604a5102f57a/src/tools/tidy/src/deps.rs#L73
