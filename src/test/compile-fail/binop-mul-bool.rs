// xfail-stage0
// error-pattern:* cannot be applied to type `bool`

fn main() {
  auto x = true * false;
}