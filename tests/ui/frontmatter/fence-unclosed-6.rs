---
//~^ ERROR unclosed frontmatter
🦀---
 ---

// This test checks the location of the --- recovered by the parser is not
// incorrectly tracked during the less fortunate recovery case and multiple
// candidates are found, as seen with #146847

#![feature(frontmatter)]

fn main() {}
