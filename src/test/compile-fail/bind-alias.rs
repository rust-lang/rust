// xfail-stage0
// error-pattern: binding alias slot

fn f(&int x) {}

fn main() {
  bind f(10);
}
