// Reported as issue #126, child leaks the string.

use std;
import std::task;

fn child2(s: -istr) { }

fn main() { let x = task::spawn(bind child2(~"hi")); }
