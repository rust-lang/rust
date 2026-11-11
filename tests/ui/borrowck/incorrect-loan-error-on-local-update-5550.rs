// https://github.com/rust-lang/rust/issues/5550
//@ run-pass
#![allow(unused_assignments)]

pub fn main() {
    let s: String = "foobar".to_string();
    let mut t: &str = &s;
    t = &t[0..3]; // for master: str::view(t, 0, 3) maybe
}
