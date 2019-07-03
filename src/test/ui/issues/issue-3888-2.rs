// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn vec_peek<'r, T>(v: &'r [T]) -> &'r [T] {
    &v[1..5]
}

pub fn main() {}
