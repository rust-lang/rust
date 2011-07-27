// error-pattern:Copying a non-copyable type

resource foo(i: int) { }

fn main() { let x <- foo(10); let y = x; }