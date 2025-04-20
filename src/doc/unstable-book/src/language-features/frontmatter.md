# `frontmatter`

The tracking issue for this feature is: [#136889]

------

The `frontmatter` feature allows an extra metadata block at the top of files for consumption by
external tools. For example, it can be used by [`cargo-script`] files to specify dependencies.

```rust
#!/usr/bin/env -S cargo -Zscript
---
[dependencies]
clap = "4"
---
#![feature(frontmatter)]

use clap::Parser;

#[derive(Parser)]
struct Cli {}

fn main() {
    Cli::parse();
}
```

[#136889]: https://github.com/rust-lang/rust/issues/136889
[`cargo-script`]: https://rust-lang.github.io/rfcs/3502-cargo-script.html
