// https://github.com/rust-lang/rust/issues/7660
//@ run-pass
#![allow(unused_variables)]
// Regression test for issue 7660
// rvalue lifetime too short when equivalent `match` works

use std::collections::HashMap;

struct A(isize, isize);

pub fn main() {
    let mut m: HashMap<isize, A> = HashMap::new();
    m.insert(1, A(0, 0));

    let A(ref _a, ref _b) = m[&1];
    let (a, b) = match m[&1] { A(ref _a, ref _b) => (_a, _b) };
}
