//@ aux-crate:reexport_self_dep=reexport_self_dep.rs
//@ edition: 2021

#![crate_name = "foo"]

//@ has 'foo/index.html' '//*[@class="item-table reexports"]' 'pub use reexport_self_dep;'
//@ !has 'foo/index.html' '//code' 'pub use self;'
pub use reexport_self_dep::{self};
