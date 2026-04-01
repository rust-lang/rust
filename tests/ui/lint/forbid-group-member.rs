// Check what happens when we forbid a group but
// then allow a member of that group.
//
//@ check-pass

#![forbid(unused)]

#[allow(unused_variables)]
//~^ WARNING incompatible with previous forbid
//~| WARNING previously accepted
fn main() {
    let a: ();
}
