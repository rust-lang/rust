mod foo {
    enum Bar {
        Baz { a: isize },
    }
}

fn f(b: foo::Bar) {
    //~^ ERROR enum `Bar` is private
    match b {
        foo::Bar::Baz { a: _a } => {} //~ ERROR enum `Bar` is private
    }
}

fn main() {}
