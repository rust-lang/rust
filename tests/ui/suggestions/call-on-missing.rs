struct Foo { i: i32 }

impl Foo {
    fn bar(&self) {}
}

fn foo() -> Foo {
    Foo { i: 1 }
}

fn main() {
    foo.bar();
    //~^ ERROR no method named `bar`
    //~| HELP use parentheses to call this function

    foo.i;
    //~^ ERROR no field `i`
    //~| HELP use parentheses to call this function

    let callable = Box::new(|| Foo { i: 1 }) as Box<dyn Fn() -> Foo>;

    callable.bar();
    //~^ ERROR no method named `bar`
    //~| HELP use parentheses to call this trait object

    callable.i;
    //~^ ERROR no field `i`
    //~| HELP use parentheses to call this trait object
}

fn type_param<T: Fn() -> Foo>(t: T) {
    t.bar();
    //~^ ERROR no method named `bar`
    //~| HELP use parentheses to call this type parameter

    t.i;
    //~^ ERROR no field `i`
    //~| HELP use parentheses to call this type parameter
}
