// -*- rust -*-

// Reported as issue #141, as a parse error. Ought to work in full though.

type list = tag(cons(int,@list), nil());
type bubu = rec(int x, int y);


fn less_than(int x, int y) -> bool { ret x < y; }

type ordered_range = rec(int low, int high) : less_than(*.low, *.high);

fn main() {
}
