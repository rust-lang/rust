

// error-pattern:wooooo
fn main() {
  auto a = 1;
  if (1 == 1) {
    a = 2;
  }
  fail "woooo" + "o";
}