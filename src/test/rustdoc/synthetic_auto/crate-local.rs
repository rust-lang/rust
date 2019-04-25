#![feature(optin_builtin_traits)]

pub auto trait Banana {}

// @has crate_local/struct.Peach.html
// @has - '//code' 'impl Banana for Peach'
// @has - '//code' 'impl Send for Peach'
// @has - '//code' 'impl Sync for Peach'
pub struct Peach;
