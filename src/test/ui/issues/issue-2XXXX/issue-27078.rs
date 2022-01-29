#![feature(associated_consts)]

trait Foo {
    const BAR: i32;
    fn foo(self) -> &'static i32 {
        //~^ ERROR the size for values of type
        &<Self>::BAR
    }
}

fn main() {}
