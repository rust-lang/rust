// error-pattern: copying a noncopyable value

resource foo(i: int) { }

fn main() { let x <- foo(10); let y = x; log_full(core::error, x); }
