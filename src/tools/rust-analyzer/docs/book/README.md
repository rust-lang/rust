# rust-analyzer documentation

The rust analyzer manual uses [mdbook](https://rust-lang.github.io/mdBook/).

## Quick start

To run the documentation site locally:

```shell
cargo install mdbook
cargo xtask codegen
cd docs/book
mdbook serve
# make changes to documentation files in doc/book/src
# ...
```

mdbook will rebuild the documentation as changes are made.

## Making updates

While not required, installing the mdbook binary can be helpful in order to see the changes.
Start with the mdbook [User Guide](https://rust-lang.github.io/mdBook/guide/installation.html) to familiarize yourself with the tool.

## Generated documentation

Four sections are generated dynamically: assists, configuration, diagnostics and features. Their content is found in the `generated.md` files
of the respective book section, for example `src/configuration_generated.md`, and are included in the book via mdbook's
[include](https://rust-lang.github.io/mdBook/format/mdbook.html#including-files) functionality. Generated files can be rebuilt by running the various
test cases that generate them, or by simply running all of the `rust-analyzer` tests with `cargo test` and `cargo xtask codegen`.
