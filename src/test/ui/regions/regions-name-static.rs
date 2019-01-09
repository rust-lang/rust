struct Foo<'static> { //~ ERROR invalid lifetime parameter name: `'static`
    x: &'static isize
}

fn main() {}
