#![allow(unused_must_use)]
#![allow(unreachable_code)]

#![allow(unused_variables)]
#![allow(dead_code)]

fn id(x: bool) -> bool { x }

fn call_id() {
    let c = panic!();
    id(c);
}

fn call_id_3() { id(return) && id(return); }

pub fn main() {
}
