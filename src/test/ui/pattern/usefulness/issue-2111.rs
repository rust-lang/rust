fn foo(a: Option<usize>, b: Option<usize>) {
  match (a,b) {
  //~^ ERROR: non-exhaustive patterns: `(None, None)` not covered
    (Some(a), Some(b)) if a == b => { }
    (Some(_), None) |
    (None, Some(_)) => { }
  }
}

fn main() {
  foo(None, None);
}
