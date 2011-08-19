// error-pattern: mismatched types

use std;
import std::task;

fn f(x: int) -> int { ret x; }

fn main() { task::_spawn(bind f(10)); }
