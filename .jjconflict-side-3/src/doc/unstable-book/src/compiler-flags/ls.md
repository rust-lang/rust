# `ls`

---

Option `-Zls` instructs the compiler to list all metadata from a given metadata file (i.e. files with the `.rmeta` extension).

This allows for debugging the metadata emitted by an earlier compilation.

Note that, while `rustc` usually works with `.rs` files, this option is meant purely for analyzing `.rmeta` files, and does not produce any compilation artifact.

Allowed values are:

- `root`: Crate info.
- `lang_items`: Language items used and missing, if any.
- `features`: Library features defined via the `#[stable]` and `#[unstable]` internal attributes.
- `items`: All items (such as modules, functions...) in the crate, including attributes like their visibility
- `target_modifiers`: Values of command-line arguments that rustc may require to match across linked crates.
- `all`: All of the above

## Example

```sh
rustc +nightly -Zls=all target/debug/deps/libmy_crate-*.rmeta
```

This lists to stdout all metadata from the given `.rmeta` file
