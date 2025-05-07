//@ run-pass
#![allow(dead_code)]

struct X { x: isize }

pub fn main() {
    let _x = match 0 {
      _ => X {
        x: 0
      }
    };
}
