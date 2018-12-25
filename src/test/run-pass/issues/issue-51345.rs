// run-pass
#![allow(unreachable_code)]
#![feature(nll)]

fn main() {
    let mut v = Vec::new();

    loop { v.push(break) }
}
