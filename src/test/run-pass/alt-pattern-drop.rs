

// -*- rust -*-
import core::sys;

enum t { make_t(@int), clam, }

fn foo(s: @int) {
    let count = sys::refcount(s);
    let x: t = make_t(s); // ref up

    match x {
      make_t(y) => {
        log(debug, y); // ref up then down

      }
      _ => { debug!{"?"}; fail; }
    }
    log(debug, sys::refcount(s));
    assert (sys::refcount(s) == count + 1u);
}

fn main() {
    let s: @int = @0; // ref up

    let count = sys::refcount(s);

    foo(s); // ref up then down

    log(debug, sys::refcount(s));
    let count2 = sys::refcount(s);
    let _ = sys::refcount(s); // don't get bitten by last-use.
    assert count == count2;
}
