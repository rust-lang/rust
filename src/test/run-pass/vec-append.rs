// -*- rust -*-

fn fast_growth() {
  let vec[int] v = vec(1,2,3,4,5);
  v += vec(6,7,8,9,0);

  log v.(9);
  check(v.(0) == 1);
  check(v.(7) == 8);
  check(v.(9) == 0);
}

fn slow_growth() {
  let vec[int] v = vec();
  let vec[int] u = v;
  v += vec(17);

  log v.(0);
  check (v.(0) == 17);
}

fn main() {
  fast_growth();
  slow_growth();
}
