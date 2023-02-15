# Clippy

[![Clippy Test](https://github.com/rust-lang/rust-clippy/workflows/Clippy%20Test%20(bors)/badge.svg?branch=auto&event=push)](https://github.com/rust-lang/rust-clippy/actions?query=workflow%3A%22Clippy+Test+(bors)%22+event%3Apush+branch%3Aauto)
[![License: MIT OR Apache-2.0](https://img.shields.io/crates/l/clippy.svg)](https://github.com/rust-lang/rust-clippy#license)

A collection of lints to catch common mistakes and improve your
[Rust](https://github.com/rust-lang/rust) code.

[There are over 600 lints included in this crate!](https://rust-lang.github.io/rust-clippy/master/index.html)

Lints are divided into categories, each with a default [lint
level](https://doc.rust-lang.org/rustc/lints/levels.html). You can choose how
much Clippy is supposed to ~~annoy~~ help you by changing the lint level by
category.

| Category              | Description                                                                         | Default level |
| --------------------- | ----------------------------------------------------------------------------------- | ------------- |
| `clippy::all`         | all lints that are on by default (correctness, suspicious, style, complexity, perf) | **warn/deny** |
| `clippy::correctness` | code that is outright wrong or useless                                              | **deny**      |
| `clippy::suspicious`  | code that is most likely wrong or useless                                           | **warn**      |
| `clippy::complexity`  | code that does something simple but in a complex way                                | **warn**      |
| `clippy::perf`        | code that can be written to run faster                                              | **warn**      |
| `clippy::style`       | code that should be written in a more idiomatic way                                 | **warn**      |
| `clippy::pedantic`    | lints which are rather strict or might have false positives                         | allow         |
| `clippy::nursery`     | new lints that are still under development                                          | allow         |
| `clippy::cargo`       | lints for the cargo manifest                                                        | allow         |                                   | allow         |

More to come, please [file an
issue](https://github.com/rust-lang/rust-clippy/issues) if you have ideas!

The [lint list](https://rust-lang.github.io/rust-clippy/master/index.html) also
contains "restriction lints", which are for things which are usually not
considered "bad", but may be useful to turn on in specific cases. These should
be used very selectively, if at all.
