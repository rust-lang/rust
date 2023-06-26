// check-pass
#![allow(incomplete_features)]
#![feature(explicit_tail_calls)]

fn _f<'a>() -> &'a [u8] {
    become _g();
}

fn _g() -> &'static [u8] {
    &[0, 1, 2, 3]
}

fn main() {}
