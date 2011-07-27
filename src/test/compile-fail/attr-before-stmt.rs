// xfail-stage0
// error-pattern:expected item

fn f() {
  #[foo = "bar"]
  let x = 10;
}

fn main() {
}
