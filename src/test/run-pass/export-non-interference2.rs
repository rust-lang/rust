mod foo {
    #[legacy_exports];

    export bar;

    mod bar {
        #[legacy_exports];
        fn y() { x(); }
    }

    fn x() { debug!("x"); }
}

fn main() { foo::bar::y(); }
