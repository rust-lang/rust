// error-pattern: mismatched types

use std;

fn main() { task::spawn(fn~() -> int { 10 }); }
