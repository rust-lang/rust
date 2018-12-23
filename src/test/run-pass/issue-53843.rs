// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ops::Deref;

pub struct Pin<P>(P);

impl<P, T> Deref for Pin<P>
where
    P: Deref<Target=T>,
{
    type Target = T;

    fn deref(&self) -> &T {
        &*self.0
    }
}

impl<P> Pin<P> {
    fn poll(self) {}
}

fn main() {
    let mut unit = ();
    let pin = Pin(&mut unit);
    pin.poll();
}
