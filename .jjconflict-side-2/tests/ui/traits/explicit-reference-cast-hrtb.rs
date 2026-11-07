//! Regression test for #158967

struct Foo;

fn f()
where
    for<'a> Foo: From<&'a String>,
{
}

fn main() {
    f();
    //~^ ERROR the trait bound `for<'a> Foo: From<&'a String>` is not satisfied [E0277]
}
