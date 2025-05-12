---
 ---
//~^ ERROR: invalid preceding whitespace for frontmatter close

 ---
//~^ ERROR: expected item, found `-`
// FIXME(frontmatter): make this diagnostic better
---

#![feature(frontmatter)]

fn main() {}
