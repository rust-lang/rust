#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

fn main() {
    // FIXME(deref_patterns): fails to typecheck because `"foo"` has type &str but deref creates a
    // place of type `str`.
    match "foo".to_string() {
        box "foo" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match &"foo".to_string() {
        box "foo" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}

struct Struct;

fn cant_move_out_box(b: Box<Struct>) -> Struct {
    match b {
        //~^ ERROR: cannot move out of a shared reference
        box x => x,
        _ => unreachable!(),
    }
}

fn cant_move_out_rc(rc: Rc<Struct>) -> Struct {
    match rc {
        //~^ ERROR: cannot move out of a shared reference
        box x => x,
        _ => unreachable!(),
    }
}
