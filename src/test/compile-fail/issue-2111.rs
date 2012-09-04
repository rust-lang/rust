fn foo(a: Option<uint>, b: Option<uint>) {
  match (a,b) { //~ ERROR: non-exhaustive patterns: None not covered
    (Some(a), Some(b)) if a == b => { }
    (Some(_), None) |
    (None, Some(_)) => { }
  }
}

fn main() {
  foo(None, None);
}