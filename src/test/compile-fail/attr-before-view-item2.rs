// xfail-stage0
// error-pattern:expected item

mod m {
  #[foo = "bar"]
  use std;
}

fn main() {
}
