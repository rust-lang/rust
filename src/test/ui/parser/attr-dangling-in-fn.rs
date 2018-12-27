// compile-flags: -Z parse-only

// error-pattern:expected statement

fn f() {
  #[foo = "bar"]
}

fn main() {
}
