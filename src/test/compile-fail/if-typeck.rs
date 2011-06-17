// xfail-stage0
// error-pattern:mismatched types
// issue #513

fn f() {}

fn main() {
  // f is not a bool
  if (f) {
  }
}