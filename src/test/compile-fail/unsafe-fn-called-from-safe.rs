// -*- rust -*-

unsafe fn f() { return; }

fn main() {
    f(); //~ ERROR access to unsafe function requires unsafe function or block
}
