// xfail-boot
fn main() {

  obj foo(mutable int i) {
      fn inc_by(int incr) -> int {
          i += incr;
          ret i;
      }

      fn inc_by_5() -> int {
          ret self.inc_by(5);
      }

      // A test case showing that issue #324 is resolved.  (It used to
      // be that commenting out this (unused!) function produced a
      // type error.)
      // fn wrapper(int incr) -> int {
      //     ret self.inc_by(incr);
      // }

      fn get() -> int {
          ret i;
      }
  }
  
  let int res;
  auto o = foo(5);
  
  res = o.get();
  assert (res == 5);

  res = o.inc_by(3);
  assert (res == 8);
  
  res = o.get();
  assert (res == 8);

}

