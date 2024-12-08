enum Foo {}

impl Foo {
    const B: u8 = 0;
}

fn bar(foo: Foo) -> u32 {
    match foo {
        Foo::B(i) => i, //~ ERROR E0164
    }
}

fn main() {
}
