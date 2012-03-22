


// -*- rust -*-
fn f(u: ()) { ret u; }

fn main() {
    let u1: () = ();
    let mut u2: () = f(u1);
    u2 = ();
    ret ();
}
