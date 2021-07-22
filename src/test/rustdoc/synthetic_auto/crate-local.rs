#![feature(auto_traits)]

pub auto trait Banana {}

// @has crate_local/struct.Peach.html
// @has - '//h3' 'impl Banana for Peach'
// @has - '//h3' 'impl Send for Peach'
// @has - '//h3' 'impl Sync for Peach'
pub struct Peach;
