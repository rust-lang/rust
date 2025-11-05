//@ edition: 2024

enum Test {
    Value = -5 >> 1_usize,
}

fn test1(x: impl Iterator<Item = Foo>) {
    //~^ ERROR cannot find type `Foo` in this scope
    assert_eq!(Test::Value as u8, -3);
}

fn test2(_: impl Iterator<Item = Foo>) {
    //~^ ERROR cannot find type `Foo` in this scope
    0u8 == -3;
    //~^ ERROR cannot apply unary operator `-` to type `u8`
}

fn main() {}
