


// error-pattern:explicit failure
fn f() -> ! { fail }

fn g() -> int { let x = alt true { true { f() } false { 10 } }; ret x; }

fn main() { g(); }
