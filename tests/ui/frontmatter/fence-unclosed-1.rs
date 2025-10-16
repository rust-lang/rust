----cargo
//~^ ERROR: unclosed frontmatter

// This test checks that the #! characters can help us recover a frontmatter
// close. There should not be a "missing `main` function" error as the rest
// are properly parsed.

//~v ERROR: `main` function not found in crate `fence_unclosed_1`
fn main() {}
