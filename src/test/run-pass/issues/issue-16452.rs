// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn main() {
    if true { return }
    match () {
        () => { static MAGIC: usize = 0; }
    }
}
