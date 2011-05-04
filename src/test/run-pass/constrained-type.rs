// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

// Reported as issue #141, as a parse error. Ought to work in full though.

tag list {
  cons(int,@list);
  nil();
}

type bubu = rec(int x, int y);


fn less_than(int x, int y) -> bool { ret x < y; }

type ordered_range = rec(int low, int high) : less_than(*.low, *.high);

fn main() {
}
