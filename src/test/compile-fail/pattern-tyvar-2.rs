// -*- rust -*-

use std;
import option;
import option::some;

// error-pattern: mismatched types

tag bar { t1((), option::t<[int]>); t2; }

fn foo(t: bar) -> int { alt t { t1(_, some(x)) { ret x * 3; } _ { fail; } } }

fn main() { }
