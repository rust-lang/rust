// xfail-boot
// xfail-stage0
// xfail-stage1
// xfail-stage2
// -*- rust -*-

tag mlist {
  cons(int,mutable @mlist);
  nil;
}

fn main() {
  cons(10, @cons(11, @cons(12, @nil)));
}
