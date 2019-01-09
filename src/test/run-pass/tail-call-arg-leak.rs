// use of tail calls causes arg slot leaks, issue #160.
// pretty-expanded FIXME #23616

fn inner(dummy: String, b: bool) { if b { return inner(dummy, false); } }

pub fn main() {
    inner("hi".to_string(), true);
}
