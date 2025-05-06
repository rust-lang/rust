  ---
//~^ ERROR: invalid preceding whitespace for frontmatter opening
  ---
//~^ ERROR: invalid preceding whitespace for frontmatter close

#![feature(frontmatter)]

// check that whitespaces should not precede the frontmatter opening or close.

fn main() {}
