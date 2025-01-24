//@ run-pass
#![allow(unreachable_code)]

fn main() {
    ({return},);
}
