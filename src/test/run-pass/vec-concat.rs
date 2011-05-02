// -*- rust -*-

fn main() {
  let vec[int] a = vec(1,2,3,4,5);
  let vec[int] b = vec(6,7,8,9,0);
  let vec[int] v = a + b;
  log v.(9);
  check(v.(0) == 1);
  check(v.(7) == 8);
  check(v.(9) == 0);
}
