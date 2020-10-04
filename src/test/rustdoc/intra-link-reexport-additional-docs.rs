#![crate_name = "foo"]

// @has foo/struct.JoinPathsError.html '//a[@href="../foo/fn.with_code.html"]' 'crate::with_code'
/// [crate::with_code]
// @has - '//a[@href="../foo/fn.with_code.html"]' 'different text'
/// [different text][with_code]
// @has - '//a[@href="../foo/fn.me_too.html"]' 'me_too'
#[doc = "[me_too]"]
// @has - '//a[@href="../foo/fn.me_three.html"]' 'reference link'
/// This [reference link]
#[doc = "has an attr in the way"]
///
/// [reference link]: me_three
pub use std::env::JoinPathsError;

pub fn with_code() {}
pub fn me_too() {}
pub fn me_three() {}
