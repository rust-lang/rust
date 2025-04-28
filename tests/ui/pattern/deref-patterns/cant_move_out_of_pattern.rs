#![feature(deref_patterns)]
#![allow(incomplete_features)]

use std::rc::Rc;

struct Struct;

fn cant_move_out_vec(b: Vec<Struct>) -> Struct {
    match b {
        //~^ ERROR: cannot move out of type `[Struct]`, a non-copy slice
        deref!([x]) => x,
        _ => panic!(),
    }
}

fn cant_move_out_rc(rc: Rc<Struct>) -> Struct {
    match rc {
        //~^ ERROR: cannot move out of a shared reference
        deref!(x) => x,
        _ => unreachable!(),
    }
}

fn cant_move_out_vec_implicit(b: Vec<Struct>) -> Struct {
    match b {
        //~^ ERROR: cannot move out of type `[Struct]`, a non-copy slice
        [x] => x,
        _ => panic!(),
    }
}

struct Container(Struct);

fn cant_move_out_rc_implicit(rc: Rc<Container>) -> Struct {
    match rc {
        //~^ ERROR: cannot move out of a shared reference
        Container(x) => x,
        _ => unreachable!(),
    }
}

fn main() {}
