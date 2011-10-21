// error-pattern: mismatched types

use std;
import std::task;

fn f(&&x: int) -> int { ret x; }

fn main() { task::spawn(10, f); }
