#![allow(unused_must_use)]
#![allow(dead_code)]

#![allow(path_statements)]
#![allow(unreachable_code)]
#![allow(unused_variables)]

fn id(x: bool) -> bool { x }

fn call_id() {
    let c = panic!();
    id(c);
}

fn call_id_2() { id(true) && id(return); }

fn call_id_3() { id(return) && id(return); }

fn ret_guard() {
    match 2 {
      x if (return) => { x; }
      _ => {}
    }
}

pub fn main() {}
