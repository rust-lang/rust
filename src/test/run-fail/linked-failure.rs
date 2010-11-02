// -*- rust -*-

// error-pattern:1 == 2

fn child() {
  check (1 == 2);
}

impure fn main() {
  let port[int] p = port();
  spawn child();
  let int x;
  x <- p;
}
