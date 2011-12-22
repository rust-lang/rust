// error-pattern: unresolved name
mod foo {
    export x;
    fn x(y: int) { log_full(core::debug, y); }
    fn z(y: int) { log_full(core::debug, y); }
}

fn main() { foo::z(10); }
