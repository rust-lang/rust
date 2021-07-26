// [feature] run-pass
// revisions: normal feature
#![cfg_attr(feature, feature(type_equality_constraints))]

struct Baz<T> {
  _t: T,
}

impl<T> Baz<T> where T = i32 {
//[normal]~^ ERROR equality constraints
  fn resolved(self) {}
}

fn main() {
  let b = Baz{_t: 1i32};
  b.resolved();
}
