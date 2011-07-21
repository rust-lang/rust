// xfail-stage0
// xfail-stage1
// xfail-stage2
// xfail-stage3
// error-pattern: binding alias slot

fn f(&int x) {}

fn main() {
  bind f(10);
}
