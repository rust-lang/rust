#![feature(nll)]

fn main() {
    let mut v = Vec::new();

    loop { v.push(break) }
}
