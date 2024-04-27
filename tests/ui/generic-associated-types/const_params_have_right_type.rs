trait Trait {
    type Foo<const N: u8>;
}

impl Trait for () {
    type Foo<const N: u64> = u32;
    //~^ error: type `Foo` has an incompatible generic parameter for trait
}

fn main() {}
