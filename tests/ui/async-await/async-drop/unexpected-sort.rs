// Ex-ice: #140484
//@ edition: 2024
#![crate_type = "lib"]
#![allow(incomplete_features)]
#![allow(non_camel_case_types)]
#![feature(async_drop)]
use std::future::AsyncDrop;
struct a;
impl Drop for a { //~ ERROR: not all trait items implemented, missing: `drop`
    fn b() {} //~ ERROR: method `b` is not a member of trait `Drop`
}
impl AsyncDrop for a { //~ ERROR: not all trait items implemented, missing: `drop`
    type c = ();
    //~^ ERROR: type `c` is not a member of trait `AsyncDrop`
}
async fn bar() {
    a;
}
