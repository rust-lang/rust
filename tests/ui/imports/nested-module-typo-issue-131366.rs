//! Regression test for https://github.com/rust-lang/rust/issues/131366.
//! Suggest accessible, similarly named modules within the resolved import prefix.

//@ edition: 2021
#![allow(unused_imports, dead_code)]

use std::collection::HashMap;
//~^ ERROR unresolved import `std::collection`

mod local {
    pub mod collections {
        pub struct Item;
    }
    pub use collections as containers;
    pub enum Choices { First }
}

use local::collection::Item;
//~^ ERROR unresolved import `local::collection`
use local::container::Item as Alias;
//~^ ERROR unresolved import `local::container`
use local::Choice::First;
//~^ ERROR unresolved import `local::Choice`

mod root_module {}
use crate::root_modul::*;
//~^ ERROR unresolved import `crate::root_modul`

macro_rules! import {
    ($module:ident) => { use local::$module::Item as MacroItem; };
}
import!(collection);
//~^ ERROR unresolved import `local::collection`

fn main() {}
