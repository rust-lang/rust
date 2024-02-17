//@ run-rustfix

pub struct Foo {
    pub first: bool,
    pub second: u8,
}

fn main() {
    let _ = Foo {
        //~^ ERROR missing field
        first: true
        second: 25
        //~^ ERROR expected one of
    };
}
