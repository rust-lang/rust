----cargo
//~^ ERROR: frontmatter close does not match the opening
---cargo
//~^ ERROR: extra characters after frontmatter close are not allowed

#![feature(frontmatter)]

fn main() {}
