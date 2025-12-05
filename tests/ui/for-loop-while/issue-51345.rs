//@ run-pass
#![allow(unreachable_code)]

fn main() {
    let mut v = Vec::new();

    loop { v.push(break) }
}
