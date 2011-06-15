


// -*- rust -*-
obj x() {
    fn hello() { log "hello, object world"; }
}

fn main() { auto mx = x(); mx.hello(); }