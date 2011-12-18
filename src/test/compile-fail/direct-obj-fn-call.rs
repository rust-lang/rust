// error-pattern: attempted access of field hello

obj x() {
    fn hello() { log "hello"; }
}

fn main() { x.hello(); }
