// -*- rust -*-

use std;
use option::Some;

// error-pattern: mismatched types

enum bar { t1((), Option<~[int]>), t2, }

fn foo(t: bar) -> int { match t { t1(_, Some(x)) => { return x * 3; } _ => { fail; } } }

fn main() { }
