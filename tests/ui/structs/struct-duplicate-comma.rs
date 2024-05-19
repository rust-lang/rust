//@ run-rustfix
// Issue #50974

pub struct Foo {
    pub a: u8,
    pub b: u8
}

fn main() {
    let _ = Foo {
        a: 0,,
          //~^ ERROR expected identifier
        b: 42
    };
}
