// error-pattern:mismatched types: expected `()` but found `bool`

struct r {
  new() {}
  drop { true }
}

fn main() {
}