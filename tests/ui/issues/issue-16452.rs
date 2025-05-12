//@ run-pass
#![allow(dead_code)]

fn main() {
    if true { return }
    match () {
        () => { static MAGIC: usize = 0; }
    }
}
