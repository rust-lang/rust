// error-pattern:expected item

mod m {
    #[legacy_exports];
  #[foo = "bar"]
  extern mod std;
}

fn main() {
}
