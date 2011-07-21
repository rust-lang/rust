// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// -*- rust -*-

// error-pattern: tag of infinite size

tag mlist {
  cons(int, mlist);
  nil;
}

fn main() {
  auto a = cons(10, cons(11, nil));
}
