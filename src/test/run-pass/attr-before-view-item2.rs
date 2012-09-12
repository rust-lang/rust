// error-pattern:expected item

mod m {
  #[foo = "bar"]
  extern mod std;
}

fn main() {
}
