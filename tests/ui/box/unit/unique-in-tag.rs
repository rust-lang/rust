//@ run-pass
#![allow(dead_code)]
#![allow(non_camel_case_types)]

fn test1() {
    enum bar { u(Box<isize>), w(isize), }

    let x = bar::u(Box::new(10));
    assert!(match x {
      bar::u(a) => {
        println!("{}", a);
        *a
      }
      _ => { 66 }
    } == 10);
}

pub fn main() {
    test1();
}
