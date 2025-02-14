# `frontmatter`

The tracking issue for this feature is: [#136889]

[#136889]: https://github.com/rust-lang/rust/issues/136889

------------------------

The `frontmatter` feature adds support for a specialized and simplified attribute syntax
intended for external tools to consume.

For example, when used with Cargo:
```rust,ignore (frontmatter/shebang are not intended for doctests)
#!/usr/bin/env -S cargo -Zscript

---
[dependencies]
clap = "4"
---

use clap::Parser;

#[derive(Parser)]
struct Cli {
}

fn main () {
    Cli::parse();
}
```

A frontmatter may come after a shebang and must come before any other syntax except whitespace.
The open delimiter is three or more dashes (`-`) at the start of a new line.
The open delimiter may be followed by whitespace and / or an identifier to mark the interpretation of the frontmatter within an external tool.
It is then concluded at a newline.
The close delimiter is a series of dashes that matches the open delimiter, at the start of a line.
The close delimiter may be followed by whitespace.
Any other trailing content, including more dashes than the open delimiter, is an error.
It is then concluded at a newline.
All content between the open and close delimiter lines is ignored.
