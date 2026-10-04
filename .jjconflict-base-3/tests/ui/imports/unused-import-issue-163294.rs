//@ edition: 2015

#![deny(unused_imports)]

pub use core as _;
//~^ ERROR unresolved import `core`
//~^^ ERROR unused import: `core as _`

fn main() {}
