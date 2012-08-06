


// error-pattern:explicit failure
fn main() { let x = match true { false => { 0 } true => { fail } }; }
