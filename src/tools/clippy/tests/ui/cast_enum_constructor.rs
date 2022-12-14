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
    let _ = Foo::Y as isize;
    let _ = Foo::Y as fn(u32) -> Foo;
    let _ = Bar::X as usize;
}
