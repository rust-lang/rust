use std::ops::Deref;

struct Foo;

impl Deref for Foo {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &[]
    }
}

fn main() {
    let _ = "foo" as bool;
    //~^ ERROR casting `&'static str` as `bool` is invalid [E0606]

    let _ = String::from("foo") as bool;
    //~^ ERROR non-primitive cast: `String` as `bool` [E0605]

    let _ = Foo as bool;
    //~^ ERROR non-primitive cast: `Foo` as `bool` [E0605]
}

fn _slice(bar: &[i32]) -> bool {
    bar as bool
    //~^ ERROR casting `&[i32]` as `bool` is invalid [E0606]
}
