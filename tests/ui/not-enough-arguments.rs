//@ aux-build: delegate_macro.rs
extern crate delegate_macro;
use delegate_macro::delegate;

// Check that the only error msg we report is the
// mismatch between the # of params, and not other
// unrelated errors.
fn foo(a: isize, b: isize, c: isize, d: isize) {
    panic!();
}

// Check that all arguments are shown in the error message, even if they're across multiple lines.
fn bar(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) {
    println!("{}", a);
    println!("{}", b);
    println!("{}", c);
    println!("{}", d);
    println!("{}", e);
    println!("{}", f);
}

macro_rules! delegate_local {
    ($method:ident) => {
        <Self>::$method(8)
        //~^ ERROR function takes 2 arguments but 1
    };
}

macro_rules! delegate_from {
    ($from:ident, $method:ident) => {
        <$from>::$method(8)
        //~^ ERROR function takes 2 arguments but 1
    };
}

struct Bar;

impl Bar {
    fn foo(a: u8, b: u8) {}
    fn bar() {
        delegate_local!(foo);
        delegate!(foo);
        //~^ ERROR function takes 2 arguments but 1
        delegate_from!(Bar, foo);
    }
}

fn main() {
    foo(1, 2, 3);
    //~^ ERROR function takes 4 arguments but 3
    bar(1, 2, 3);
    //~^ ERROR function takes 6 arguments but 3
}
