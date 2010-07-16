// -*- rust -*-

// error-pattern: impure function used in constraint

fn f(int a, int b) : lt(a,b) {
}

io fn lt(int a, int b) -> bool {
  let port[int] p = port();
  let chan[int] c = chan(p);
  c <| 10;
  ret true;
}

fn main() {
  let int a = 10;
  let int b = 23;
  check lt(a,b);
  f(a,b);
}
