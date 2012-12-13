// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Just a named container for our op, so it can have impls
pub struct NominalOp<T> {
    op: T
}

impl<T: Copy> NominalOp<T>: Clone {
    fn clone(&self) -> NominalOp<T> { copy *self }
}

pub fn spawn_listener<A: Owned>(
    +f: fn~(comm::Port<A>)) -> comm::Chan<A> {
    let setup_po = comm::Port();
    let setup_ch = comm::Chan(&setup_po);
    do task::spawn |move f| {
        let po = comm::Port();
        let ch = comm::Chan(&po);
        comm::send(setup_ch, ch);
        f(move po);
    }
    comm::recv(setup_po)
}

pub fn spawn_conversation<A: Owned, B: Owned>
    (+f: fn~(comm::Port<A>, comm::Chan<B>))
    -> (comm::Port<B>, comm::Chan<A>) {
    let from_child = comm::Port();
    let to_parent = comm::Chan(&from_child);
    let to_child = do spawn_listener |move f, from_parent| {
        f(from_parent, to_parent)
    };
    (from_child, to_child)
}
