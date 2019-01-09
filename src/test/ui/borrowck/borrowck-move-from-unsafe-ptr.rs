unsafe fn foo(x: *const Box<isize>) -> Box<isize> {
    let y = *x; //~ ERROR cannot move out of dereference of raw pointer
    return y;
}

fn main() {
}
