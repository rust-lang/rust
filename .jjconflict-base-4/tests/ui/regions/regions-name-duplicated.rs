struct Foo<'a, 'a> {
    //~^ ERROR the name `'a` is already used for a generic parameter
    x: &'a isize,
}

fn main() {}
