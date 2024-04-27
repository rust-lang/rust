//@ aux-build:re-export.rs
//@ edition:2018

extern crate re_export;

use re_export::cause_ice;

cause_ice!(); //~ ERROR `Variant` is only public within the crate, and cannot be re-exported outside

fn main() {}
