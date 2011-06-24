// xfail-stage0
// error-pattern:&& cannot be applied to type `int`

fn main() {
  auto x = 1 && 2;
}