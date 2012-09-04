// error-pattern: copying a noncopyable value

struct foo {
  let i: int;
  new(i:int) { self.i = i; }
  drop {}
}

fn main() { let x <- foo(10); let y = x; log(error, x); }
