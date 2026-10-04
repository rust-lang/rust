#![feature(auto_traits)]

pub auto trait Banana {}

//@ has crate_local/struct.Peach.html
//@ has - '//h3[@class="code-header"]' 'impl Banana for Peach'
//@ has - '//h3[@class="code-header"]' 'impl Send for Peach'
//@ has - '//h3[@class="code-header"]' 'impl Sync for Peach'
pub struct Peach;
