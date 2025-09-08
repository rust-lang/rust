// https://github.com/rust-lang/rust/issues/5741
//@ run-pass
#![allow(while_true)]
#![allow(unreachable_code)]

pub fn main() {
    return;
    while true {};
}
