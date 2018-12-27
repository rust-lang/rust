// compile-flags: -Z parse-only

// error-pattern: expected

fn main() {
  let isize x = 5;
  match x;
}

fn main() {
}
