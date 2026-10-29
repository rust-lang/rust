Rustdoc will automatically scrape examples of documented items from a project's source code. These examples will be included within the generated documentation for that item. For example, if your library contains a public function:

```rust
// src/lib.rs
pub fn a_func() {}
```

And you have an example calling this function:

```rust
// examples/ex.rs
fn main() {
  a_crate::a_func();
}
```

Then this code snippet will be included in the documentation for `a_func`.


## How to read scraped examples

Scraped examples are shown as blocks of code from a given file. The relevant item will be highlighted. If the file is larger than a couple lines, only a small window will be shown which you can expand by clicking &varr; in the top-right. If a file contains multiple instances of an item, you can use the &pr; and &sc; buttons to toggle through each instance.

If there is more than one file that contains examples, then you should click "More examples" to see these examples.


## How Rustdoc scrapes examples

When you run `cargo doc -Zunstable-options -Zrustdoc-scrape-examples`, Rustdoc will analyze all the documented crates for uses of documented items. Then Rustdoc will include the source code of these instances in the generated documentation.

Rustdoc has a few techniques to ensure this doesn't overwhelm documentation readers, and that it doesn't blow up the page size:

1. For a given item, a maximum of 5 examples are included in the page. The remaining examples are just links to source code.
2. Only one example is shown by default, and the remaining examples are hidden behind a toggle.
3. For a given file that contains examples, only the item containing the examples will be included in the generated documentation.
