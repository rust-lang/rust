#![feature(arbitrary_enum_discriminant)]

#[repr(u8)]
enum Kind2 {
    Foo() = 1,
    Bar{} = 2,
    Baz = 3,
}

fn main() {
    let _ = Kind2::Foo() as u8;
    //~^ ERROR non-primitive cast
}
