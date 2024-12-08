//@ run-pass

#![allow(unused_assignments)]
#![allow(unknown_lints)]

#![allow(unused_variables)]
#![allow(dead_assignment)]

fn f(u: ()) { return u; }

pub fn main() {
    let u1: () = ();
    let mut u2: () = f(u1);
    u2 = ();
    return ();
}
