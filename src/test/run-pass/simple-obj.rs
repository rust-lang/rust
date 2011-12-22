


// -*- rust -*-
obj x() {
    fn hello() { #debug("hello, object world"); }
}

fn main() { let mx = x(); mx.hello(); }
