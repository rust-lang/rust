trait Foo {}

impl<F> Foo for F where F: Fn(&i32) -> &i32 {}

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(|a| a); //~ ERROR mismatched types
    take_foo(|a: &i32| a); //~ ERROR mismatched types
    take_foo(|a: &i32| -> &i32 { a }); //~ ERROR mismatched types

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
