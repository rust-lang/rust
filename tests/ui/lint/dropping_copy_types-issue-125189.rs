//@ check-fail
//@ run-rustfix

#![deny(dropping_copy_types)]

#[allow(unused_variables)]
fn main() {
    let y = 1;
    drop(3.2); //~ ERROR calls to `std::mem::drop`
    drop(y); //~ ERROR calls to `std::mem::drop`
}
