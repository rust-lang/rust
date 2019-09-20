// run-pass

// FIXME: this should not pass

#![feature(const_fn)]
#![feature(const_fn_ptr)]

fn double(x: usize) -> usize { x * 2 }
const X: fn(usize) -> usize = double;

const fn bar(x: usize) -> usize {
    X(x)
}

fn main() {}
