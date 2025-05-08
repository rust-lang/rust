//@ no-prefer-dynamic

#![crate_type = "rlib"]

#[macro_use] #[no_link] extern crate empty_crate_1 as t1;
#[macro_use] extern crate empty_crate_2 as t2;
