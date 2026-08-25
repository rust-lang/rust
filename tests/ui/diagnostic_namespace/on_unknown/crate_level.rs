//@ compile-flags: --crate-name crate_level_test
//@ edition: 2018..
// See https://doc.rust-lang.org/nightly/reference/items/use-declarations.html#r-items.use.path.edition2018

#![feature(custom_inner_attributes)]
#![feature(diagnostic_on_unknown)]
#![crate_type = "lib"]

// Only applies to the root, not everything in the entire crate
#![diagnostic::on_unknown(message = "crate_level `{This}`")]

mod module{}

pub use crate::foo;
//~^ ERROR crate_level `crate_level_test`

pub use ::bar;
//~^ ERROR unresolved import `bar`

pub use baz;
//~^ ERROR unresolved import `baz`

pub use crate::module::waz;
//~^ ERROR unresolved import `crate::module::waz`

pub use module::bluz;
//~^ ERROR unresolved import `module::bluz`
