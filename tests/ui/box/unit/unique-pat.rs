//@ run-pass

#![feature(box_patterns)]

fn simple() {
    match Box::new(true) {
      box true => { }
      _ => { panic!(); }
    }
}

pub fn main() {
    simple();
}
