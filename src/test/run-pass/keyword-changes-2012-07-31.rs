#![allow(dead_code)]
// return -> return
// mod -> module
// match -> match

// pretty-expanded FIXME #23616

pub fn main() {
}

mod foo {
}

fn bar() -> isize {
    match 0 {
      _ => { 0 }
    }
}
