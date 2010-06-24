// -*- rust -*-

fn f(int x) -> int {
  if (x == 1) {
    ret 1;
  } else {
    let int y = 1 + f(x-1);
    ret y;
  }
}

fn main() {
  check (f(5000) == 5000);
}
