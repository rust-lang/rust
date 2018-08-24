#![feature(rustc_attrs)]

use std::ops::Deref;

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let _x: fn(&i32) -> <&i32 as Deref>::Target = unimplemented!();
}
