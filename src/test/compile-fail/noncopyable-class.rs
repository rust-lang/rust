// error-pattern: copying a noncopyable value

// Test that a class with a non-copyable field can't be
// copied
class bar {
  new() {}
  drop {}
}

class foo {
  let i: int;
  let j: bar;
  new(i:int) { self.i = i; self.j = bar(); }
}

fn main() { let x <- foo(10); let y = x; log(error, x); }
