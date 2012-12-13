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
    +f: fn~(oldcomm::Port<A>)) -> oldcomm::Chan<A> {
    let setup_po = oldcomm::Port();
    let setup_ch = oldcomm::Chan(&setup_po);
    do task::spawn |move f| {
        let po = oldcomm::Port();
        let ch = oldcomm::Chan(&po);
        oldcomm::send(setup_ch, ch);
        f(move po);
    }
    oldcomm::recv(setup_po)
}

pub fn spawn_conversation<A: Owned, B: Owned>
    (+f: fn~(oldcomm::Port<A>, oldcomm::Chan<B>))
    -> (oldcomm::Port<B>, oldcomm::Chan<A>) {
    let from_child = oldcomm::Port();
    let to_parent = oldcomm::Chan(&from_child);
    let to_child = do spawn_listener |move f, from_parent| {
        f(from_parent, to_parent)
    };
    (from_child, to_child)
}
