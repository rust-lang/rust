// check-pass
#![feature(negative_impls)]

struct Foo<T>(T);

impl !Send for Foo<()> {}

fn test<T>() -> T where Foo<T>: Send { todo!() }

fn main() {
    let _: u8 = test();
}
