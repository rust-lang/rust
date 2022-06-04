#![allow(clippy::needless_late_init, clippy::manual_swap)]
#![allow(unused_variables, unused_assignments)]
#![warn(clippy::almost_swapped)]

fn main() {
    let b = 1;
    let a = b;
    let b = a;

    let mut c = 1;
    let mut d = 2;
    d = c;
    c = d;

    let mut b = 1;
    let a = b;
    b = a;

    let b = 1;
    let a = 2;

    let t = b;
    let b = a;
    let a = t;

    let mut b = 1;
    let mut a = 2;

    let t = b;
    b = a;
    a = t;
}
