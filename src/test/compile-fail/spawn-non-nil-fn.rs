// error-pattern: mismatched types

use std;
import task;

fn main() { task::spawn(fn~() -> int { 10 }); }
