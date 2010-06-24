// -*- rust -*-

// error-pattern: Infinite type recursion

type mlist = tag(cons(int,mlist), nil());

fn main() {
  auto a = cons(10, cons(11, nil()));
}
