use std::fmt::Debug;
use std::ptr;
use std::rc::Rc;
use std::sync::Arc;

fn a() {}

#[warn(clippy::fn_address_comparisons)]
fn main() {
    type F = fn();
    let f: F = a;
    let g: F = f;

    // These should fail:
    let _ = f == a;
    //~^ ERROR: comparing with a non-unique address of a function item
    //~| NOTE: `-D clippy::fn-address-comparisons` implied by `-D warnings`
    let _ = f != a;
    //~^ ERROR: comparing with a non-unique address of a function item

    // These should be fine:
    let _ = f == g;
}
