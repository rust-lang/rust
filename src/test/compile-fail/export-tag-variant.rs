// error-pattern: unresolved name

mod foo {
    export x;

    fn x() { }

    tag y { y1; }
}

fn main() { let z = foo::y1; }
