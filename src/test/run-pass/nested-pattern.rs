

// a bug was causing this to complain about leaked memory on exit
use std;
import option;
import option::some;
import option::none;

tag t { foo(int, uint); bar(int, option::t<int>); }

fn nested(o: t) {
    alt o {
      bar(i, some::<int>(_)) { log_err "wrong pattern matched"; fail; }
      _ { log_err "succeeded"; }
    }
}

fn main() { nested(bar(1, none::<int>)); }
