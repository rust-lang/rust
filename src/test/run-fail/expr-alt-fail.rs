


// error-pattern:explicit failure
fn main() { let x = alt true { false { 0 } true { fail } }; }
