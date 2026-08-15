---cargo
//~^ ERROR: frontmatter close does not match the opening
----

// there must be the same number of dashes for both the opening and the close
// of the frontmatter.

#![feature(frontmatter)]

fn main() {}
