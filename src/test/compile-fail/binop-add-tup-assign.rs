// xfail-stage0
// error-pattern:+ cannot be applied to type `tup(bool)`

fn main() {
  auto x = tup(true);
  x += tup(false);
}