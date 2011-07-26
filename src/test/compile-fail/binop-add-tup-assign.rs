// xfail-stage0
// error-pattern:+ cannot be applied to type `rec(bool x)`

fn main() {
  auto x = rec(x=true);
  x += rec(x=false);
}