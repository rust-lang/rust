# Clippy

[### IMPORTANT NOTE FOR CONTRIBUTORS ================](development/feature_freeze.md)

----

[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/clippy.svg)](https://github.com/rust-lang/rust-clippy#license)

A collection of lints to catch common mistakes and improve your
[Rust](https://github.com/rust-lang/rust) code.

[There are over 750 lints included in this crate!](https://rust-lang.github.io/rust-clippy/master/index.html)

Lints are divided into categories, each with a default [lint
level](https://doc.rust-lang.org/rustc/lints/levels.html). You can choose how
much Clippy is supposed to ~~annoy~~ help you by changing the lint level by
category.

| Category              | Description                                                                         | Default level |
|-----------------------|-------------------------------------------------------------------------------------|---------------|
| `clippy::all`         | all lints that are on by default (correctness, suspicious, style, complexity, perf) | **warn/deny** |
| `clippy::correctness` | code that is outright wrong or useless                                              | **deny**      |
| `clippy::suspicious`  | code that is most likely wrong or useless                                           | **warn**      |
| `clippy::style`       | code that should be written in a more idiomatic way                                 | **warn**      |
| `clippy::complexity`  | code that does something simple but in a complex way                                | **warn**      |
| `clippy::perf`        | code that can be written to run faster                                              | **warn**      |
| `clippy::pedantic`    | lints which are rather strict or have occasional false positives                    | allow         |
| `clippy::restriction` | lints which prevent the use of language and library features[^restrict]             | allow         |
| `clippy::nursery`     | new lints that are still under development                                          | allow         |
| `clippy::cargo`       | lints for the cargo manifest                                                        | allow         |

More to come, please [file an issue](https://github.com/rust-lang/rust-clippy/issues) if you have ideas!

The `restriction` category should, *emphatically*, not be enabled as a whole. The contained
lints may lint against perfectly reasonable code, may not have an alternative suggestion,
and may contradict any other lints (including other categories). Lints should be considered
on a case-by-case basis before enabling.

[^restrict]: Some use cases for `restriction` lints include:
    - Strict coding styles (e.g. [`clippy::else_if_without_else`]).
    - Additional restrictions on CI (e.g. [`clippy::todo`]).
    - Preventing panicking in certain functions (e.g. [`clippy::unwrap_used`]).
    - Running a lint only on a subset of code (e.g. `#[forbid(clippy::float_arithmetic)]` on a module).

[`clippy::else_if_without_else`]: https://rust-lang.github.io/rust-clippy/master/index.html#else_if_without_else
[`clippy::todo`]: https://rust-lang.github.io/rust-clippy/master/index.html#todo
[`clippy::unwrap_used`]: https://rust-lang.github.io/rust-clippy/master/index.html#unwrap_used
