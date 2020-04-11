// run-pass

#![feature(destructuring_assignment)]

fn main() {
  let (mut a, mut b);
  [a, b] = [0, 1];
  assert_eq!((a,b), (0,1));
  [a, .., b] = [1,2];
  assert_eq!((a,b), (1,2));
}
