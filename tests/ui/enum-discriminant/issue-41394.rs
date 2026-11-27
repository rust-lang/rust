enum Foo {
    A = "" + 1
    //~^ ERROR: can't add `{integer}` to `&str`
}

enum Bar {
    A = Foo::A as isize
}

fn main() {}
