// error-pattern: copying a noncopyable value

// Test that a class with a non-copyable field can't be
// copied
struct bar {
  let x: int;
  new(x:int) {self.x = x;}
  drop {}
}

struct foo {
  let i: int;
  let j: bar;
  new(i:int) { self.i = i; self.j = bar(5); }
}

fn main() { let x <- foo(10); let y = x; log(error, x); }
