#![feature(auto_traits)]

pub auto trait Banana {}

// @has crate_local/struct.Peach.html
// @has - '//h3[@class="code-header in-band"]' 'impl Banana for Peach'
// @has - '//h3[@class="code-header in-band"]' 'impl Send for Peach'
// @has - '//h3[@class="code-header in-band"]' 'impl Sync for Peach'
pub struct Peach;
