#![warn(clippy::cast_enum_constructor)]
#![allow(clippy::fn_to_numeric_cast)]

fn main() {
    enum Foo {
        Y(u32),
    }

    enum Bar {
        X,
    }

    let _ = Foo::Y as usize;
    //~^ ERROR: cast of an enum tuple constructor to an integer
    //~| NOTE: `-D clippy::cast-enum-constructor` implied by `-D warnings`
    let _ = Foo::Y as isize;
    //~^ ERROR: cast of an enum tuple constructor to an integer
    let _ = Foo::Y as fn(u32) -> Foo;
    let _ = Bar::X as usize;
}
