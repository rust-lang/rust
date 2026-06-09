# Scraped examples

Rustdoc has an unstable feature where it can automatically scrape examples of items being documented from the `examples/` directory of a Cargo workspace. These examples will be included within the generated documentation for that item. For example, if your library contains a public function:

```rust,ignore (needs-other-file)
// a_crate/src/lib.rs
pub fn a_func() {}
```

And you have an example calling this function:

```rust,ignore (needs-other-file)
// a_crate/examples/ex.rs
fn main() {
  a_crate::a_func();
}
```

Then this code snippet will be included in the documentation for `a_func`. This documentation is inserted by Rustdoc and cannot be manually edited by the crate author.


## How to use this feature

This feature is unstable, so you can enable it by calling Rustdoc with the unstable `rustdoc-scrape-examples` flag:

```bash
cargo doc -Zunstable-options -Zrustdoc-scrape-examples
```

To enable this feature on [docs.rs](https://docs.rs), add this to your Cargo.toml:

```toml
[package.metadata.docs.rs]
cargo-args = ["-Zunstable-options", "-Zrustdoc-scrape-examples"]
```


## How it works

When you run `cargo doc`, Rustdoc will analyze all the crates that match Cargo's `--examples` filter for instances of items being documented. Then Rustdoc will include the source code of these instances in the generated documentation.

Rustdoc has a few techniques to ensure these examples don't overwhelm documentation readers, and that it doesn't blow up the page size:

1. For a given item, a maximum of 5 examples are included in the page. The remaining examples are just links to source code.
2. Only one example is shown by default, and the remaining examples are hidden behind a toggle.
3. For a given file that contains examples, only the item containing the examples will be included in the generated documentation.

For a given item, Rustdoc sorts its examples based on the size of the example &mdash; smaller ones are shown first.


## FAQ

### My example is not showing up in the documentation

This feature uses Cargo's convention for finding examples. You should ensure that `cargo check --examples` includes your example file.
