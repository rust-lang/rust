// -*- rust -*-

type colour = tag(red(int,int), green());

fn f() {
  auto x = red(1,2);
  auto y = green();
  // FIXME: needs structural equality test working.
  // check (x != y);
}

fn main() {
  f();
}
