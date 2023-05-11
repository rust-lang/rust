trait Foo {
    fn foo(_: fn(u8) -> ());
    fn bar(_: Option<u8>);
    fn baz(_: (u8, u16));
    fn qux() -> u8;
}

struct Bar;

impl Foo for Bar {
    fn foo(_: fn(u16) -> ()) {}
    //~^ ERROR method `foo` has an incompatible type for trait
    fn bar(_: Option<u16>) {}
    //~^ ERROR method `bar` has an incompatible type for trait
    fn baz(_: (u16, u16)) {}
    //~^ ERROR method `baz` has an incompatible type for trait
    fn qux() -> u16 { 5u16 }
    //~^ ERROR method `qux` has an incompatible type for trait
}

fn main() {}
