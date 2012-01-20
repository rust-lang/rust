// error-pattern: unresolved name

mod foo {
    export x;

    fn x() { }

    enum y { y1; }
}

fn main() { let z = foo::y1; }
