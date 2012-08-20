

// a bug was causing this to complain about leaked memory on exit
use std;
import option;
import option::Some;
import option::None;

enum t { foo(int, uint), bar(int, Option<int>), }

fn nested(o: t) {
    match o {
      bar(i, Some::<int>(_)) => { error!("wrong pattern matched"); fail; }
      _ => { error!("succeeded"); }
    }
}

fn main() { nested(bar(1, None::<int>)); }
