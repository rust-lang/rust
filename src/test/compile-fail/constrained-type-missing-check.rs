// -*- rust -*-
// xfail-stage0
// xfail-stage1
// xfail-stage2
// error-pattern:Unsatisfied precondition

tag list {
  cons(int,@list);
  nil();
}

type bubu = rec(int x, int y);

pred less_than(int x, int y) -> bool { ret x < y; }

type ordered_range = rec(int low, int high) : less_than(*.low, *.high);

fn main() {
// Should fail to compile, b/c we're not doing the check
// explicitly that a < b
  let int a = 1;
  let int b = 2;
  let ordered_range c = rec(low=a, high=b);
  log c.low;
}
