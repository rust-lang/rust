// Test that hidden_glob_reexports lint doesn't warn when importing the same item
// that's already available through a glob re-export

pub use std::option::*;
use std::option::Option; // Should not warn - same item as glob import

fn main() {
    let _x: Option<i32> = Some(42);
}
