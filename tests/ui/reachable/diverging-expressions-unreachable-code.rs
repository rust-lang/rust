//@ run-pass

#![allow(unused_must_use)]
#![allow(unreachable_code)]

fn _id(x: bool) -> bool {
    x
}

fn _call_id() {
    let _c = panic!();
    _id(_c);
}

fn _call_id_3() {
    _id(return) && _id(return);
}

pub fn main() {}
