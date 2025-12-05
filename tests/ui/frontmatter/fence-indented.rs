  ---
//~^ ERROR: invalid preceding whitespace for frontmatter opening
//~^^ ERROR: unclosed frontmatter
  ---

#![feature(frontmatter)]

// check that whitespaces should not precede the frontmatter opening or close.

fn main() {}
