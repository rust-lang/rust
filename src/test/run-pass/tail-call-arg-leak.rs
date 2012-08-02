


// use of tail calls causes arg slot leaks, issue #160.
fn inner(dummy: ~str, b: bool) { if b { return inner(dummy, false); } }

fn main() { inner(~"hi", true); }
