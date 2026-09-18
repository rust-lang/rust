// issue#140796

trait Bar {
    fn method() -> impl Sized;
    fn method() -> impl Sized;  //~ ERROR: the name `method` is defined multiple times
}

impl Bar for () {               //~ ERROR: not all trait items implemented, missing: `method`
    fn method() -> impl Sized {
        42
    }
    fn method() -> impl Sized { //~ ERROR: duplicate definitions with name `method`
        42
    }
}

trait T {
    fn method() -> impl Sized;
}

impl T for () {
    fn method() -> impl Sized {
        42
    }
    fn method() -> impl Sized { //~ ERROR: duplicate definitions with name `method`
        42
    }
}

trait Baz {
    fn foo();
    fn foo() -> impl Sized;     //~ ERROR: the name `foo` is defined multiple times
}

trait Foo {
    fn foo() -> impl Sized;
    fn foo();                   //~ ERROR: the name `foo` is defined multiple times
    fn foo() -> impl Sized;     //~ ERROR: the name `foo` is defined multiple times
}

fn main() {}
