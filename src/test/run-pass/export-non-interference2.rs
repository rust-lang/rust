mod foo {

    export bar;

    mod bar {
        fn y() { x(); }
    }

    fn x() { log "x"; }
}

fn main() { foo::bar::y(); }
