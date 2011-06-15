

mod foo {
    mod bar {
        fn y() { x(); }
    }
    fn x() { log "x"; }
}

fn main() { foo::bar::y(); }