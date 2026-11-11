enum Foo {
    A = "" + 1
    //~^ ERROR cannot add `{integer}` to `&str`
}

enum Bar {
    A = Foo::A as isize
}

fn main() {}
