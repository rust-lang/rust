// xfail-boot
fn main() {

  obj foo() {
      fn m1(mutable int i) -> int {
          i += 1;
          ret i;
      }
      fn m2(mutable int i) -> int {
          ret self.m1(i);
      }
      fn m3(mutable int i) -> int {
          i += 1;
          ret self.m1(i);
      }
  }
  
  auto a = foo();
  let int i = 0;

  i = a.m1(i);
  check (i == 1);
  i = a.m2(i);
  check (i == 2);
  i = a.m3(i);
  check (i == 4);
}


