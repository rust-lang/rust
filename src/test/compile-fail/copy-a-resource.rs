// error-pattern:Copying a non-copyable type

// This is still not properly checked
// xfail-test

resource foo(i: int) { }

fn main() { let x <- foo(10); let y = x; }
