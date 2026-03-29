// frontmatters must be at the start of a file. This test ensures that.

---
//~^ ERROR: expected item, found `-`
// FIXME(frontmatter): make this diagnostic better
---

fn main() {}
