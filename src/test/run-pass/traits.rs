// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//xfail-test

// Sketching traits.

// methods with no implementation are required; methods with an
// implementation are provided.  No "req" keyword necessary.
trait Eq {
    fn eq(a: self) -> bool;

    fn neq(a: self) -> bool {
        !self.eq(a)
    }
}

// The `<` is pronounced `extends`.  Also under consideration is `<:`.
// Just using `:` is frowned upon, because (paraphrasing dherman) `:`
// is supposed to separate things from different universes.
trait Ord < Eq {

    fn lt(a: self) -> bool;

    fn lte(a: self) -> bool {
        self.lt(a) || self.eq(a)
    }

    fn gt(a: self) -> bool {
        !self.lt(a) && !self.eq(a)
    }

    fn gte(a: self) -> bool {
        !self.lt(a)
    }
}

// pronounced "impl of Ord for int" -- not sold on this yet
impl Ord for int {
    fn lt(a: &int) -> bool {
        self < (*a)
    }

    // is this the place to put this?
    fn eq(a: &int) -> bool {
        self == (*a)
    }
}
