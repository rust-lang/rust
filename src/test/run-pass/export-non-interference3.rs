mod foo {
    #[legacy_exports];
    export x;

    fn x() { bar::x(); }
}

mod bar {
    #[legacy_exports];
    export x;

    fn x() { debug!("x"); }
}

fn main() { foo::x(); }
