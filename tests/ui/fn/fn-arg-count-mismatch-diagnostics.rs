//! Checks clean diagnostics for argument count mismatches without unrelated errors.
//!
//! `delegate!` part related: <https://github.com/rust-lang/rust/pull/140591>

//@ aux-build: delegate_macro.rs
extern crate delegate_macro;
use delegate_macro::delegate;

fn foo(a: isize, b: isize, c: isize, d: isize) {
    panic!();
}

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

fn function_with_lots_of_arguments(a: i32, b: char, c: i32, d: i32, e: i32, f: i32) {}

fn main() {
    foo(1, 2, 3);
    //~^ ERROR function takes 4 arguments but 3
    bar(1, 2, 3);
    //~^ ERROR function takes 6 arguments but 3

    let variable_name = 42;
    function_with_lots_of_arguments(
        variable_name,
        variable_name,
        variable_name,
        variable_name,
        variable_name,
    );
    //~^^^^^^^ ERROR this function takes 6 arguments but 5 arguments were supplied [E0061]
}
