// run-pass
#![feature(const_fn)]
#![feature(const_fn_ptr)]

const fn double(x: usize) -> usize { x * 2 }
const X: fn(usize) -> usize = double;

const fn bar(x: usize) -> usize {
    X(x)
}

const fn foo(x: fn(usize) -> usize, y: usize)  -> usize {
    x(y)
}

fn main() {
    const Y: usize = bar(2);
    assert_eq!(Y, 4);
    const Z: usize = foo(double, 2);
    assert_eq!(Z, 4);
}
