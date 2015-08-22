// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct ForceCloneFrom;
struct ForceClone;

impl Clone for ForceCloneFrom {
    fn clone(&self) -> Self {
        // Ensure `clone_from` is called
        assert!(false);
        ForceCloneFrom
    }
    fn clone_from(&mut self, _: &Self) {
    }
}

impl Clone for ForceClone {
    fn clone(&self) -> Self {
        ForceClone
    }
    fn clone_from(&mut self, _: &Self) {
        // Ensure `clone` is called
        assert!(false);
    }
}

#[derive(Clone)]
enum E {
    A(ForceCloneFrom),
    B(ForceClone),
}

pub fn main() {
    let mut a = E::A(ForceCloneFrom);
    let b = E::A(ForceCloneFrom);
    let c = E::B(ForceClone);

    // This should use `clone_from` internally because they are
    // the same variant
    a.clone_from(&b);

    // This should use `clone` internally because they are different
    // variants
    a.clone_from(&c);
}
