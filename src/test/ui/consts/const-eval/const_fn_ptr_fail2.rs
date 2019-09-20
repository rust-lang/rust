#![feature(const_fn)]
#![feature(const_fn_ptr)]

fn double(x: usize) -> usize { x * 2 }
const X: fn(usize) -> usize = double;

const fn bar(x: fn(usize) -> usize, y: usize) -> usize {
    x(y)
}

const Y: usize = bar(X, 2);
//~^ ERROR any use of this value will cause an error

const Z: usize = bar(double, 2);
//~^ ERROR any use of this value will cause an error


fn main() {
    assert_eq!(Y, 4);
    assert_eq!(Z, 4);
}
