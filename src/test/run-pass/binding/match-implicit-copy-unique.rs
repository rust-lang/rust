// run-pass
#![allow(non_shorthand_field_patterns)]
#![feature(box_syntax)]

struct Pair { a: Box<isize>, b: Box<isize> }

pub fn main() {
    let mut x: Box<_> = box Pair {a: box 10, b: box 20};
    let x_internal = &mut *x;
    match *x_internal {
      Pair {a: ref mut a, b: ref mut _b} => {
        assert_eq!(**a, 10);
        *a = box 30;
        assert_eq!(**a, 30);
      }
    }
}
