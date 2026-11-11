unsafe fn foo(x: *const Box<isize>) -> Box<isize> {
    let y = *x; //~ ERROR cannot move out of `*x` which is behind a raw pointer
    return y;
}

fn main() {
}
