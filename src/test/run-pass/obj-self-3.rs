// xfail-boot
fn main() {

  obj foo() {
      impure fn m1(mutable int i) -> int {
          i += 1;
          ret i;
      }
      impure fn m2(mutable int i) -> int {
          ret self.m1(i);
      }
      impure fn m3(mutable int i) -> int {
          i += 1;
          ret self.m1(i);
      }
  }
  
  auto a = foo();
  let int i = 0;

  // output should be: 0, 1, 2, 4
  log i;
  i = a.m1(i);
  log i;
  i = a.m2(i);
  log i;
  i = a.m3(i);
  log i;
}


