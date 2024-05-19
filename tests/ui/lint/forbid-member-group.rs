// Check what happens when we forbid a member of
// a group but then allow the group.

#![forbid(unused_variables)]

#[allow(unused)]
//~^ ERROR incompatible with previous forbid
fn main() {
    let a: ();
}
