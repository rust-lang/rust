// error-pattern:cannot copy pinned type foo

resource foo(i: int) { }

fn main() { let x <- foo(10); let y = x; }
