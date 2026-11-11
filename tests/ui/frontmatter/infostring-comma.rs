---cargo,clippy
//~^ ERROR: invalid infostring for frontmatter
---

// infostrings can only be a single identifier.

#![feature(frontmatter)]

fn main() {}
