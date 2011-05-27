// -*- rust -*-
// error-pattern:src/test/compile-fail/shadow.rs
fn foo(vec[int] c) {
  let int a = 5;
  let vec[int] b = [];

  alt (none[int]) {
    case (some[int](_)) {
      for (int i in c) {
        log a;
        auto a = 17;
        b += [a];
      }
    }
  }
}

tag t[T] {
  none;
  some(T);
}

fn main() {
  foo([]);
}
