// Reported as issue #126, child leaks the string.

use std;
import std::task;

fn child2(&&s: str) { }

fn main() { let x = task::spawn("hi", child2); }
