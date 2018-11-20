// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// pretty-expanded FIXME #23616

enum Void {}

fn void() -> Void {
    unimplemented!()
}

fn void_to_string() -> String {
    match void() {}
}

pub fn main() {}
