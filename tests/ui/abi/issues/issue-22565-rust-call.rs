#![feature(unboxed_closures)]

extern "rust-call" fn b(_i: i32) {}
//~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument

trait Tr {
    extern "rust-call" fn a();
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument

    extern "rust-call" fn b() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}

struct Foo;

impl Foo {
    extern "rust-call" fn bar() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}

impl Tr for Foo {
    extern "rust-call" fn a() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}

fn main() {
    b(10);
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    Foo::bar();
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    <Foo as Tr>::a();
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
    <Foo as Tr>::b();
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self tuple argument
}
