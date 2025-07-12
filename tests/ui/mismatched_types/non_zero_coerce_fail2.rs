use std::num::NonZero;

trait Foo<T>: Sized {
    fn foo(&self, other: &T) {}
    fn bar(self, other: T) {}
}

impl Foo<u8> for u8 {}
impl Foo<u16> for u16 {}

fn main() {
    let x = NonZero::new(5_u8).unwrap();
    5_u8.foo(&x);
    //~^ ERROR: mismatched types
    5_u8.bar(x);
    5.foo(&x);
    //~^ ERROR: the trait bound `{integer}: Foo<NonZero<u8>>` is not satisfied
    5.bar(x);
    //~^ ERROR: the trait bound `{integer}: Foo<NonZero<u8>>` is not satisfied

    let a = NonZero::new(5).unwrap();
    5_u8.foo(&a);
    //~^ ERROR: mismatched types
    5_u8.bar(a);
    5.foo(&a);
    //~^ ERROR: the trait bound `{integer}: Foo<NonZero<u8>>` is not satisfied
    5.bar(a);
    //~^ ERROR: the trait bound `{integer}: Foo<NonZero<u8>>` is not satisfied
}
