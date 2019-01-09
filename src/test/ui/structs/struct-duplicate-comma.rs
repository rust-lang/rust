// Issue #50974

struct Foo {
    a: u8,
    b: u8
}

fn main() {
    let bar = Foo {
        a: 0,,
          //~^ ERROR expected identifier
        b: 42
    };
}
