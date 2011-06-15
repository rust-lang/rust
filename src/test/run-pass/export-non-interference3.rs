

mod foo {
    fn x() { bar::x(); }
}

mod bar {
    fn x() { log "x"; }
}

fn main() { foo::x(); }