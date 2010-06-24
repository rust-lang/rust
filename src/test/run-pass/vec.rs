// -*- rust -*-

fn main() {
  let vec[int] v = vec(10, 20);
  check (v.(0) == 10);
  check (v.(1) == 20);
  let int x = 0;
  check (v.(x) == 10);
  check (v.(x + 1) == 20);
  x = x + 1;
  check (v.(x) == 20);
  check (v.(x-1) == 10);
}
