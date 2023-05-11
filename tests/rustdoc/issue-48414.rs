// aux-build:issue-48414.rs

// ICE when resolving paths for a trait that linked to another trait, when both were in an external
// crate

#![crate_name = "base"]

extern crate issue_48414;

#[doc(inline)]
pub use issue_48414::{SomeTrait, OtherTrait};
