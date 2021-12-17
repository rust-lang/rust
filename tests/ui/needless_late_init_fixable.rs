// run-rustfix

#![allow(unused, clippy::assign_op_pattern)]

fn main() {
    let a;
    a = "zero";

    let b;
    let c;
    b = 1;
    c = 2;

    let d: usize;
    d = 1;

    let mut e;
    e = 1;
    e = 2;

    let h;
    h = format!("{}", e);

    println!("{}", a);
}
