//@ revisions: next old
//@[next] compile-flags: -Znext-solver
trait Foo {}

impl<F> Foo for F where F: Fn(&i32) -> &i32 {}

fn take_foo(_: impl Foo) {}

fn main() {
    take_foo(|a| a);
    //[old]~^ ERROR: implementation of `FnOnce` is not general enough
    //[old]~| ERROR: implementation of `Fn` is not general enough
    //[next]~^^^ ERROR: the trait bound `{closure@$DIR/issue-79187-2.rs:10:14: 10:17}: Foo` is not satisfied
    take_foo(|a: &i32| a);
    //[old]~^ ERROR: lifetime may not live long enough
    //[old]~| ERROR: mismatched types
    //[next]~^^^ ERROR: type mismatch
    take_foo(|a: &i32| -> &i32 { a });
    //[old]~^ ERROR: lifetime may not live long enough
    //[old]~| ERROR: mismatched types
    //[next]~^^^ ERROR: type mismatch

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
