mod foo {

    export bar;

    mod bar {
        fn y() { x(); }
    }

    fn x() { #debug("x"); }
}

fn main() { foo::bar::y(); }
