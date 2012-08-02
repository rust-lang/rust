// -*- rust -*-

unsafe fn f() { return; }

fn main() {
    let x = f; //~ ERROR access to unsafe function requires unsafe function or block
    x();
}
