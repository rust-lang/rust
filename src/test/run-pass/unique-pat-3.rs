#![feature(box_syntax)]

enum bar { u(Box<isize>), w(isize), }

pub fn main() {
    assert!(match bar::u(box 10) {
      bar::u(a) => {
        println!("{}", a);
        *a
      }
      _ => { 66 }
    } == 10);
}
