fn foo(a: option<uint>, b: option<uint>) {
  alt (a,b) { //! ERROR: non-exhaustive patterns: none not covered
    (some(a), some(b)) if a == b { }
    (some(_), none) |
    (none, some(_)) { }
  }
}

fn main() {
  foo(none, none);
}