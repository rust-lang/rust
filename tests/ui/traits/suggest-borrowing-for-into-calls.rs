struct Foo;
impl From<u64> for Foo {
    fn from(_: u64) -> Foo {
        Foo
    }
}

fn foo(_: &Foo) {}
fn foo_mut(_: &mut Foo) {}

fn main() {
    foo(42u64.into());
    //~^ ERROR the trait bound `&Foo: From<u64>` is not satisfied
    //~| HELP the trait `From<u64>` is implemented for `Foo`
    //~| HELP consider borrowing here
    //~| SUGGESTION &

    let val = 42u64.into();
    foo(val);
    //~^^ ERROR the trait bound `&Foo: From<u64>` is not satisfied
    //~| HELP the trait `From<u64>` is implemented for `Foo`
    //~| HELP consider borrowing here
    //~| SUGGESTION &

    foo_mut(42u64.into());
    //~^ ERROR the trait bound `&mut Foo: From<u64>` is not satisfied
    //~| HELP the trait `From<u64>` is implemented for `Foo`
    //~| HELP consider borrowing here
    //~| SUGGESTION &mut

    foo(42i32.into());
    //~^ ERROR the trait bound `&Foo: From<i32>` is not satisfied
    //~| HELP the trait `From<u64>` is implemented for `Foo`
}
