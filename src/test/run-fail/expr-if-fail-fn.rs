


// error-pattern:explicit failure
fn f() -> ! { fail }

fn g() -> int { let x = if true { f() } else { 10 }; return x; }

fn main() { g(); }
