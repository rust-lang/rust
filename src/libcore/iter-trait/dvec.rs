// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod inst {
    use dvec;
    use option::{Option, Some};

    #[allow(non_camel_case_types)]
    pub type IMPL_T<A> = dvec::DVec<A>;

    /**
    * Iterates through the current contents.
    *
    * Attempts to access this dvec during iteration will fail.
    */
    pub pure fn EACH<A>(self: &IMPL_T<A>, f: fn(v: &A) -> bool) {
        unsafe {
            do self.swap |v| {
                v.each(f);
                move v
            }
        }
    }

    pub pure fn SIZE_HINT<A>(self: &IMPL_T<A>) -> Option<uint> {
        Some(self.len())
    }
}
