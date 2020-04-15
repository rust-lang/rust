// run-pass

#![feature(destructuring_assignment)]

fn main() {
  let (mut a, mut b);
  [a, b] = [0, 1];
  assert_eq!((a, b), (0, 1));
  let mut c;
  [a, .., b, c] = [1, 2, 3, 4, 5];
  assert_eq!((a, b, c), (1, 4, 5));
  [_, a, _] = [1, 2, 3];
  assert_eq!((a, b), (2, 4));
  [..] = [1, 2, 3];
  [c, ..] = [5, 6, 6];
  assert_eq!(c, 5);
}
