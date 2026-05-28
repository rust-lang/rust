#![warn(clippy::tabs_in_doc_comments)]

///
/// Struct to hold two strings:
/// 	- first		one
//~^ tabs_in_doc_comments
//~| tabs_in_doc_comments
/// 	- second	one
//~^ tabs_in_doc_comments
//~| tabs_in_doc_comments
pub struct DoubleString {
    ///
    /// 	- First String:
    //~^ tabs_in_doc_comments
    /// 		- needs to be inside here
    //~^ tabs_in_doc_comments
    first_string: String,
    ///
    /// 	- Second String:
    //~^ tabs_in_doc_comments
    /// 		- needs to be inside here
    //~^ tabs_in_doc_comments
    second_string: String,
}

/// This is main
fn main() {}
