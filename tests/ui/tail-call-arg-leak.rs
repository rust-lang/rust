//@ run-pass
// use of tail calls causes arg slot leaks, issue #160.

fn inner(dummy: String, b: bool) { if b { return inner(dummy, false); } }

pub fn main() {
    inner("hi".to_string(), true);
}
