// error-pattern: mismatched types

use std;
import task;

fn main() { task::spawn(sendfn() -> int { 10 }); }
