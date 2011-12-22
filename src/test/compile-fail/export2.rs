// error-pattern: unresolved name

mod foo {
    export x;

    fn x() { bar::x(); }
}

mod bar {
    export y;

    fn x() { #debug("x"); }

    fn y() { }
}

fn main() { foo::x(); }
