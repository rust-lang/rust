// xfail-stage0
// error-pattern:meep
fn f(int a, int b, @int c) {
  fail "moop";
}

fn main() {
  f(1, fail "meep", @42);
}