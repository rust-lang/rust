// -*- rust -*-

unsafe fn f() { ret; }

fn main() {
    f(); //~ ERROR access to unsafe function requires unsafe function or block
}
