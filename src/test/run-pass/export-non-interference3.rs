mod foo {
    export x;

    fn x() { bar::x(); }
}

mod bar {
    export x;

    fn x() { log "x"; }
}

fn main() { foo::x(); }