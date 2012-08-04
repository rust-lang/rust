// -*- rust -*-

use std;
import option;
import option::some;

// error-pattern: mismatched types

enum bar { t1((), option<~[int]>), t2, }

fn foo(t: bar) -> int { alt t { t1(_, some(x)) => { return x * 3; } _ => { fail; } } }

fn main() { }
