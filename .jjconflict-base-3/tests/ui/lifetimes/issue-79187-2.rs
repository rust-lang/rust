trait Foo {}

impl<F> Foo for F where F: Fn(&i32) -> &i32 {}

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(|a| a);
    //~^ ERROR implementation of `FnOnce` is not general enough
    //~| ERROR implementation of `Fn` is not general enough
    take_foo(|a: &i32| a);
    //~^ ERROR lifetime may not live long enough
    //~| ERROR mismatched types
    take_foo(|a: &i32| -> &i32 { a });
    //~^ ERROR lifetime may not live long enough
    //~| ERROR mismatched types

    // OK
    take_foo(identity(|a| a));
    take_foo(identity(|a: &i32| a));
    take_foo(identity(|a: &i32| -> &i32 { a }));

    fn identity<F>(t: F) -> F
    where
        F: Fn(&i32) -> &i32,
    {
        t
    }
}
