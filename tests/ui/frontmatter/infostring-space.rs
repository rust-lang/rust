--- cargo clippy
//~^ ERROR: invalid infostring for frontmatter
---

// infostrings cannot have spaces

#![feature(frontmatter)]

fn main() {}
