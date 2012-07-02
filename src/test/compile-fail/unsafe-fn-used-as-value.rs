// -*- rust -*-

unsafe fn f() { ret; }

fn main() {
    let x = f; //~ ERROR access to unsafe function requires unsafe function or block
    x();
}
