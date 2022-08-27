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
}
