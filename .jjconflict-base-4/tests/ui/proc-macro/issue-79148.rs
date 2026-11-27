//@ proc-macro: re-export.rs
//@ edition:2018
//@ ignore-backends: gcc

extern crate re_export;

use re_export::cause_ice;

cause_ice!(); //~ ERROR `Variant` is only public within the crate, and cannot be re-exported outside

fn main() {}
