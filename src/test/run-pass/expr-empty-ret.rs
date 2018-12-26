#![allow(dead_code)]
// Issue #521

// pretty-expanded FIXME #23616

fn f() {
    let _x = match true {
        true => { 10 }
        false => { return }
    };
}

pub fn main() { }
