//@ build-pass

trait FooTrait {}

trait BarTrait {
    fn foo<T: FooTrait>(_: T) -> Self;
}

struct FooStruct(u32);

impl BarTrait for FooStruct {
    fn foo<T: FooTrait>(_: T) -> Self {
        Self(u32::default())
    }
}

fn main() {}
