// test that defaulted const params are not used to help type inference

struct Foo<const N: u32 = 2>;

impl<const N: u32> Foo<N> {
    fn foo() -> Self {
        loop {}
    }
}

fn main() {
    let foo = Foo::<1>::foo();
    let foo = Foo::foo();
    //~^ ERROR type annotations needed for `Foo<_>`
    //~| ERROR type annotations needed
}
