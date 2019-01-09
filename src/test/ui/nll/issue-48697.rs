// Regression test for #48697

// compile-pass

#![feature(nll)]

fn foo(x: &i32) -> &i32 {
    let z = 4;
    let f = &|y| y;
    let k = f(&z);
    f(x)
}

fn main() {}
