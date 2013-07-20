// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::sync::UnsafeAtomicRcBox;


/// An atomically reference counted wrapper for shared immutable state.
pub struct Arc<T> { priv contents: UnsafeAtomicRcBox<T> }

impl <T:Freeze + Send> Arc<T> {
    /// Create an atomically reference counted wrapper.
    #[inline]
    pub fn new(value: T) -> Arc<T> {
        Arc { contents: UnsafeAtomicRcBox::new(value) }
    }

    /**
    * Access the underlying data in an atomically reference counted
    * wrapper.
    */
    #[inline]
    pub fn get<'r>(&'r self) -> &'r T { unsafe { &*self.contents.get_immut() } }
}

/**
 * Duplicate an atomically reference counted wrapper.
 *
 * The resulting two `arc` objects will point to the same underlying data
 * object. However, one of the `arc` objects can be sent to another task,
 * allowing them to share the underlying data.
*/
impl<T:Freeze + Send> Clone for Arc<T> {
    #[inline]
    fn clone(&self) -> Arc<T> {
        Arc { contents: self.contents.clone() }
    }
}


#[test]
fn manually_share_arc() {
    use std::comm;
    use std::task;

    let v = ~[1u, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let arc_v = Arc::new(v);

    let (p, c) = comm::stream();

    do task::spawn {
        let p = comm::PortSet::new();
        c.send(p.chan());

        let arc_v: Arc<~[uint]> = p.recv();

        let v = arc_v.get().clone();
        assert_eq!(v[3], 4);
    };

    let c = p.recv();
    c.send(arc_v.clone());

    assert_eq!(arc_v.get()[2], 3);
    assert_eq!(arc_v.get()[4], 5);

    info!(arc_v);
}