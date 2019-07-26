// run-pass
// pretty-expanded FIXME #23616

#![allow(unused_variables)]

pub fn main() {
    let mut i: isize = 0;
    while i < 1000000 {
        i += 1;
        let x = 3;
    }
}
