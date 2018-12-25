// rustc-env:RUST_NEW_ERROR_FORMAT

#![feature(const_fn)]

trait Foo {
    fn f() -> u32;
    const fn g(); //~ ERROR cannot be declared const
}

impl Foo for u32 {
    const fn f() -> u32 { 22 } //~ ERROR cannot be declared const
    fn g() {}
}

fn main() { }
