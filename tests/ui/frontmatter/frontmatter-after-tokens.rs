#![feature(frontmatter)]

---
//~^ ERROR: expected item, found `-`
// FIXME(frontmatter): make this diagnostic better
---

// frontmatters must be at the start of a file. This test ensures that.

fn main() {}
