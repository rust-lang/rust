// error-pattern:mismatched kinds

resource foo(i: int) { }

fn main() { let x <- foo(10); let y = x; }
