struct Point {
    x: isize,
    y: isize,
}

fn a() {
    let mut p = vec![1];

    // Create an immutable pointer into p's contents:
    let q: &isize = &p[0];

    p[0] = 5; //~ ERROR cannot borrow

    println!("{}", *q);
}

fn borrow<F>(_x: &[isize], _f: F) where F: FnOnce() {}

fn b() {
    // here we alias the mutable vector into an imm slice and try to
    // modify the original:

    let mut p = vec![1];

    borrow(
        &p,
        || p[0] = 5); //~ ERROR cannot borrow `p` as mutable
}

fn c() {
    // Legal because the scope of the borrow does not include the
    // modification:
    let mut p = vec![1];
    borrow(&p, ||{});
    p[0] = 5;
}

fn main() {
}
