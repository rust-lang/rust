


// -*- rust -*-
obj x() {
    fn hello() { log "hello, object world"; }
}

fn main() { let mx = x(); mx.hello(); }
