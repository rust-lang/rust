// run-pass
#![allow(unreachable_code)]

fn main() {
    let mut v: Vec<()> = Vec::new();

    loop { v.push(break) }
}
