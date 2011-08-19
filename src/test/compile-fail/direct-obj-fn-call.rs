// error-pattern: attempted field access

obj x() {
    fn hello() { log "hello"; }
}

fn main() { x.hello(); }
