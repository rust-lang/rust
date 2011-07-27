


// error-pattern:explicit failure
fn f() -> ! { fail }

fn g() -> int { let x = if true { f() } else { 10 }; ret x; }

fn main() { g(); }