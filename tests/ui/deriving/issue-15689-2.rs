// check-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

#[derive(Clone)]
enum Test<'a> {
    Slice(&'a isize)
}

fn main() {}
