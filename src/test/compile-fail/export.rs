// error-pattern: unresolved name
mod foo {
    export x;
    fn x(y: int) { log y; }
    fn z(y: int) { log y; }
}

fn main() { foo::z(10); }