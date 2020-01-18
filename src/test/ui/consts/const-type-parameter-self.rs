// check-pass

struct Foo;

impl Foo {
    const A: usize = 37;

    fn bar() -> [u8; Self::A]{
        [0; Self::A]
    }
}

fn main() {
    let _: [u8; 37] = Foo::bar();
}
