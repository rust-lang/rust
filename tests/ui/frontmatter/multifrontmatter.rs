---
---

---
//~^ ERROR: expected item, found `-`
// FIXME(frontmatter): make this diagnostic better
---

// test that we do not parse another frontmatter block after the first one.

#![feature(frontmatter)]

fn main() {}
