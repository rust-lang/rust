// -*- rust -*-
use std;
import option;
import option::some;

// error-pattern: mismatched types

tag bar { t1((), option::t<[int]>); t2; }

fn foo(t: bar) { alt t { t1(_, some::<int>(x)) { log x; } _ { fail; } } }

fn main() { }
