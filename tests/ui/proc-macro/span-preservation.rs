// For each of these, we should get the appropriate type mismatch error message,
// and the function should be echoed.

//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

#[recollect_attr]
fn a() {
    let x: usize = "hello"; //~ ERROR mismatched types
}

#[recollect_attr]
fn b(x: Option<isize>) -> usize {
    match x {
        Some(x) => { return x }, //~ ERROR mismatched types
        None => 10
    }
}

#[recollect_attr]
fn c() {
    struct Foo {
        a: usize
    }

    struct Bar {
        a: usize,
        b: usize
    }

    let x = Foo { a: 10isize }; //~ ERROR mismatched types
    let y = Foo { a: 10, b: 10isize }; //~ ERROR has no field named `b`
}

#[recollect_attr]
extern "C" fn bar() {
    0 //~ ERROR mismatched types
}

#[recollect_attr]
extern "C" fn baz() {
    0 //~ ERROR mismatched types
}

#[recollect_attr]
extern "Rust" fn rust_abi() {
    0 //~ ERROR mismatched types
}

#[recollect_attr]
extern "\x43" fn c_abi_escaped() {
    0 //~ ERROR mismatched types
}

fn main() {}
