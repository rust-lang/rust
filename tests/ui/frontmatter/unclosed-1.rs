----cargo
//~^ ERROR: unclosed frontmatter

// This test checks that the #! characters can help us recover a frontmatter
// close. There should not be a "missing `main` function" error as the rest
// are properly parsed.

fn main() {}  //~ ERROR: `main` function not found in crate `unclosed_1` [E0601]
