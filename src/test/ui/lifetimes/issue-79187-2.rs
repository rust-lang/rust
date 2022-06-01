// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

trait Foo {}

impl<F> Foo for F where F: Fn(&i32) -> &i32 {}

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(|a| a);
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR implementation of `FnOnce` is not general enough
    //[nll]~| ERROR mismatched types
    take_foo(|a: &i32| a);
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
    //[nll]~| ERROR mismatched types
    take_foo(|a: &i32| -> &i32 { a });
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
    //[nll]~| ERROR mismatched types

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
