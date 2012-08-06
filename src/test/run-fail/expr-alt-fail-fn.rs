


// error-pattern:explicit failure
fn f() -> ! { fail }

fn g() -> int { let x = match true { true => { f() } false => { 10 } }; return x; }

fn main() { g(); }
