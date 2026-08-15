----cargo
//~^ ERROR: unclosed frontmatter

//! Similarly, a module-level content should allow for recovery as well (as
//! per unclosed-1.rs)

#![feature(frontmatter)]

fn main() {}
