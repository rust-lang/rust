//! Check that `super` keyword used at the crate root (top-level) results in a compilation error
//! as there is no parent module to resolve.

use super::f; //~ ERROR there are too many leading `super` keywords

fn main() {}
