// error-pattern: unresolved name

mod foo {
    #[legacy_exports];
    export x;

    fn x() { bar::x(); }
}

mod bar {
    #[legacy_exports];
    export y;

    fn x() { debug!("x"); }

    fn y() { }
}

fn main() { foo::x(); }
