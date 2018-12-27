enum Foo {
    A = "" + 1
    //~^ ERROR binary operation `+` cannot be applied to type `&str`
}

enum Bar {
    A = Foo::A as isize
    //~^ ERROR evaluation of constant value failed
}

fn main() {}
