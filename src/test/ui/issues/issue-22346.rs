// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

// This used to cause an ICE because the retslot for the "return" had the wrong type
fn testcase<'a>() -> Box<dyn Iterator<Item=usize> + 'a> {
    return Box::new((0..3).map(|i| { return i; }));
}

fn main() {
}
