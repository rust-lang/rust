// error-pattern: attempted access of field hello

obj x() {
    fn hello() { #debug("hello"); }
}

fn main() { x.hello(); }
