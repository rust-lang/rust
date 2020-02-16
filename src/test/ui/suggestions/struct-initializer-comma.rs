struct Foo {
    first: bool,
    second: u8,
}

fn main() {
    let a = Foo {
        //~^ ERROR missing field
        first: true
        second: 25
        //~^ ERROR expected one of
    };
}
