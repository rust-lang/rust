//! Checks the basic usage of unit type

//@ check-pass

fn f(u: ()) {
    u
}

pub fn main() {
    let u1: () = ();
    let mut _u2: () = f(u1);
    _u2 = ();
    ()
}
