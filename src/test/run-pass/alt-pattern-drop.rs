

// -*- rust -*-
use std;

import std::dbg;

tag t { make_t(@int); clam; }

fn foo(s: @int) {
    let count = dbg::refcount(s);
    let x: t = make_t(s); // ref up

    alt x {
      make_t(y) {
        log(debug, y); // ref up then down

      }
      _ { #debug("?"); fail; }
    }
    log(debug, dbg::refcount(s));
    assert (dbg::refcount(s) == count + 1u);
}

fn main() {
    let s: @int = @0; // ref up

    foo(s); // ref up then down

    log(debug, dbg::refcount(s));
    assert (dbg::refcount(s) == 1u);
}
