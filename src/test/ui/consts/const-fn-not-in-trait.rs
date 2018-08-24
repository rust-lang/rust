// Test that const fn is illegal in a trait declaration, whether or
// not a default is provided.

#![feature(const_fn)]

trait Foo {
    const fn f() -> u32;
    //~^ ERROR trait fns cannot be declared const
    const fn g() -> u32 { 0 }
    //~^ ERROR trait fns cannot be declared const
}

fn main() { }
