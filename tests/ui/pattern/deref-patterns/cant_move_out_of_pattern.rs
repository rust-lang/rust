#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

struct Struct;

fn cant_move_out_box(b: Box<Struct>) -> Struct {
    match b {
        //~^ ERROR: cannot move out of a shared reference
        deref!(x) => x,
        _ => unreachable!(),
    }
}

fn cant_move_out_rc(rc: Rc<Struct>) -> Struct {
    match rc {
        //~^ ERROR: cannot move out of a shared reference
        deref!(x) => x,
        _ => unreachable!(),
    }
}

struct Container(Struct);

fn cant_move_out_box_implicit(b: Box<Container>) -> Struct {
    match b {
        //~^ ERROR: cannot move out of a shared reference
        Container(x) => x,
        _ => unreachable!(),
    }
}

fn cant_move_out_rc_implicit(rc: Rc<Container>) -> Struct {
    match rc {
        //~^ ERROR: cannot move out of a shared reference
        Container(x) => x,
        _ => unreachable!(),
    }
}

fn main() {}
