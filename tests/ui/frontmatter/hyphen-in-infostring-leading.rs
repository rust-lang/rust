--- -toml
//~^ ERROR: invalid infostring for frontmatter
---

// infostrings cannot have leading hyphens

#![feature(frontmatter)]

fn main() {}
