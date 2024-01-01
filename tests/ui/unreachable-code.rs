// build-pass



#![allow(path_statements)]
#![allow(unreachable_code)]

#![feature(if_let_guard)]

fn id(x: bool) -> bool {
    x
}

fn call_id() {
    let c = panic!();
    id(c);
}

fn call_id_2() {
    id(true) && id(return);
}

fn call_id_3() {
    id(return) && id(return);
}

fn ret_guard() {
    match 2 {
      x if (return) => { x; }
      x if let true = return => { x; }
      _ => {}
    }
}

pub fn main() {}
