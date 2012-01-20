// -*- rust -*-
use std;
import option;
import option::some;

// error-pattern: mismatched types

enum bar { t1((), option::t<[int]>), t2, }

fn foo(t: bar) {
    alt t {
      t1(_, some::<int>(x)) {
        log(debug, x);
      }
      _ { fail; }
    }
}

fn main() { }
