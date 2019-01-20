// aux-build:issue-57676.rs

extern crate issue_57676;

// This tests that when assembling the extension candidates for traits, the presence of a trait
// alias from another crate will not cause an ICE (the actual errors produced by this once the
// ICE is fixed aren't interesting).

use std::path::Path;

fn main() {
    let d: Path = Path::new(".");
    //~^ ERROR mismatched types [E0308]
    //~^^ ERROR the size for values of type `[u8]` cannot be known at compilation time [E0277]
}
