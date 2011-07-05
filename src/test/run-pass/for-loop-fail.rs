// xfail-stage0
fn main() {
  let vec[int] x = [];
  for (int i in x) {
    fail "moop";
  }
}
