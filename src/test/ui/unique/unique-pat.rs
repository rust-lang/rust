// run-pass

#![feature(box_patterns)]
#![feature(box_syntax)]

fn simple() {
    match box true {
      box true => { }
      _ => { panic!(); }
    }
}

pub fn main() {
    simple();
}
