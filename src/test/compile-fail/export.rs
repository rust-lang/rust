// error-pattern: unresolved name
mod foo {
    export x;
    fn x(y: int) { log(debug, y); }
    fn z(y: int) { log(debug, y); }
}

fn main() { foo::z(10); }
