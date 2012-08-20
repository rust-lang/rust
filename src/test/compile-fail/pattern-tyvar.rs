// -*- rust -*-
use std;
import option;
import option::Some;

// error-pattern: mismatched types

enum bar { t1((), Option<~[int]>), t2, }

fn foo(t: bar) {
    match t {
      t1(_, Some::<int>(x)) => {
        log(debug, x);
      }
      _ => { fail; }
    }
}

fn main() { }
