mod foo {
    export x;

    fn x() { bar::x(); }
}

mod bar {
    export x;

    fn x() { #debug("x"); }
}

fn main() { foo::x(); }
