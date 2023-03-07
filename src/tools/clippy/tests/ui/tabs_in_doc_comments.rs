// run-rustfix

#![warn(clippy::tabs_in_doc_comments)]
#[allow(dead_code)]

///
/// Struct to hold two strings:
/// 	- first		one
/// 	- second	one
pub struct DoubleString {
    ///
    /// 	- First String:
    /// 		- needs to be inside here
    first_string: String,
    ///
    /// 	- Second String:
    /// 		- needs to be inside here
    second_string: String,
}

/// This is main
fn main() {}
