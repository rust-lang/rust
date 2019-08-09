// run-pass

#![deny(dead_code)]

pub struct GenericFoo<T>(T);

type Foo = GenericFoo<u32>;

impl Foo {
    fn bar(self) -> u8 {
        0
    }
}

fn main() {
    println!("{}", GenericFoo(0).bar());
}
