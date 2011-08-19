


// error-pattern:explicit failure
fn main() { let x = if false { 0 } else if true { fail } else { 10 }; }
