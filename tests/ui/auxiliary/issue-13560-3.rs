// no-prefer-dynamic

#![crate_type = "rlib"]

#[macro_use] #[no_link] extern crate issue_13560_1 as t1;
#[macro_use] extern crate issue_13560_2 as t2;
