// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

fn f() {
   let _x: usize = loop { loop { break; } };
}

pub fn main() {
}
