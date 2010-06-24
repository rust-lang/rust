// -*- rust -*-

type pair = rec(int head, mutable @mlist tail);
type mlist = tag(cons(@pair), nil());

fn main() {
  let @pair p = rec(head=10, tail=mutable nil());
  let @mlist cycle = cons(p);
  //p.tail = cycle;
}
