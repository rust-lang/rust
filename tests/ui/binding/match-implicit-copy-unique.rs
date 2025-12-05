//@ run-pass
#![allow(non_shorthand_field_patterns)]

struct Pair { a: Box<isize>, b: Box<isize> }

pub fn main() {
    let mut x: Box<_> = Box::new(Pair { a: Box::new(10), b: Box::new(20) });
    let x_internal = &mut *x;
    match *x_internal {
      Pair {a: ref mut a, b: ref mut _b} => {
        assert_eq!(**a, 10);
        *a = Box::new(30);
        assert_eq!(**a, 30);
      }
    }
}
