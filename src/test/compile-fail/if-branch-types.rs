// error-pattern:mismatched types

fn main() { let x = if true { 10 } else { 10u }; }