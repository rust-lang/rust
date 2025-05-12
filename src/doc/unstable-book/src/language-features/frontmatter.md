# `frontmatter`

The tracking issue for this feature is: [#136889]

------

The `frontmatter` feature allows an extra metadata block at the top of files for consumption by
external tools. For example, it can be used by [`cargo-script`] files to specify dependencies.

```rust
#!/usr/bin/env -S cargo -Zscript
---
[dependencies]
libc = "0.2.172"
---
#![feature(frontmatter)]
# mod libc { pub type c_int = i32; }

fn main() {
    let x: libc::c_int = 1i32;
}
```

[#136889]: https://github.com/rust-lang/rust/issues/136889
[`cargo-script`]: https://rust-lang.github.io/rfcs/3502-cargo-script.html
