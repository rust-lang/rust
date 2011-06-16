// xfail-stage0
// error-pattern:expected item

fn f() {
  #[foo = "bar"]
  auto x = 10;
}

fn main() {
}
