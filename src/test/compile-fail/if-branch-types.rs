// error-pattern:mismatched types

fn main() { let x = if true { 10i } else { 10u }; }
