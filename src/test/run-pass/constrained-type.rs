// xfail-stage0
// -*- rust -*-

tag list {
  cons(int,@list);
  nil();
}

type bubu = rec(int x, int y);

pred less_than(int x, int y) -> bool { ret x < y; }

type ordered_range = rec(int low, int high) : less_than(*.low, *.high);

fn main() {
}
