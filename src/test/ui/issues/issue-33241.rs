#![feature(rustc_attrs)]

use std::fmt;

// CoerceUnsized is not implemented for tuples. You can still create
// an unsized tuple by transmuting a trait object.
fn any<T>() -> T { unreachable!() }

#[rustc_error]
fn main() { //~ ERROR compilation successful
    let t: &(u8, fmt::Debug) = any();
    println!("{:?}", &t.1);
}
