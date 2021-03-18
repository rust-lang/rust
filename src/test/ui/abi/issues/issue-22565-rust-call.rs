#![feature(unboxed_closures)]

extern "rust-call" fn b(_i: i32) {}
//~^ ERROR functions with the "rust-call" ABI must take a single non-self argument that is a tuple

trait Tr {
    extern "rust-call" fn a();

    extern "rust-call" fn b() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self argument
}

struct Foo;

impl Foo {
    extern "rust-call" fn bar() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self argument
}

impl Tr for Foo {
    extern "rust-call" fn a() {}
    //~^ ERROR functions with the "rust-call" ABI must take a single non-self argument
}

fn main () {
    b(10);

    Foo::bar();

    <Foo as Tr>::a();
    <Foo as Tr>::b();
}
