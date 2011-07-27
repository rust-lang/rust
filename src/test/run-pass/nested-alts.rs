
use std;
import std::option::*;

fn baz() -> ! { fail; }

fn foo() {
    alt some[int](5) {
      some[int](x) {
        let bar;
        alt none[int] { none[int]. { bar = 5; } _ { baz(); } }
        log bar;
      }
      none[int]. { log "hello"; }
    }
}

fn main() { foo(); }