// xfail-boot
fn main() {

  obj foo() {
      impure fn m1(mutable int i) {
          i += 1;
          log "hi!";
      }
      impure fn m2(mutable int i) {
          i += 1;
          self.m1(i);
      }
  }
  
  auto a = foo();
  let int i = 0;
  a.m1(i);
  a.m2(i);
}
