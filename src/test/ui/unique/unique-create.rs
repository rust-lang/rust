// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub fn main() {
    let _: Box<_> = Box::new(100);
}

fn vec() {
    vec![0];
}
