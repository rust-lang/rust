use std::num::NonZero;

trait Foo: Sized {
    fn foo(&self) {}
    fn bar(self) {}
}

impl Foo for u8 {}
impl Foo for u16 {}

fn foo(_: impl Foo) {}

fn main() {
    let x = NonZero::new(5_u8).unwrap();
    x.foo();
    //~^ ERROR: no method named `foo` found for struct `NonZero` in the current scope
    x.bar();
    //~^ ERROR: no method named `bar` found for struct `NonZero` in the current scope
    foo(x);
    //~^ ERROR: the trait bound `NonZero<u8>: Foo` is not satisfied
    foo(x as _);

    let a = NonZero::new(5).unwrap();
    a.foo();
    //~^ ERROR: no method named `foo` found for struct `NonZero` in the current scope
    a.bar();
    //~^ ERROR: no method named `bar` found for struct `NonZero` in the current scope
    foo(a);
    //~^ ERROR: the trait bound `NonZero<{integer}>: Foo` is not satisfied
    foo(a as _);
}
