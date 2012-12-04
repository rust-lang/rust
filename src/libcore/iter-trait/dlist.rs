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
    #[allow(non_camel_case_types)]
    pub type IMPL_T<A> = dlist::DList<A>;

    /**
    * Iterates through the current contents.
    *
    * Attempts to access this dlist during iteration are allowed (to
    * allow for e.g. breadth-first search with in-place enqueues), but
    * removing the current node is forbidden.
    */
    pub pure fn EACH<A>(self: &IMPL_T<A>, f: fn(v: &A) -> bool) {
        let mut link = self.peek_n();
        while option::is_some(&link) {
            let nobe = option::get(link);
            assert nobe.linked;
            if !f(&nobe.data) { break; }
            // Check (weakly) that the user didn't do a remove.
            if self.size == 0 {
                fail ~"The dlist became empty during iteration??"
            }
            if !nobe.linked ||
                (!((nobe.prev.is_some()
                    || managed::ptr_eq(*self.hd.expect(~"headless dlist?"),
                                   *nobe))
                   && (nobe.next.is_some()
                    || managed::ptr_eq(*self.tl.expect(~"tailless dlist?"),
                                   *nobe)))) {
                fail ~"Removing a dlist node during iteration is forbidden!"
            }
            link = nobe.next_link();
        }
    }

    pub pure fn SIZE_HINT<A>(self: &IMPL_T<A>) -> Option<uint> {
        Some(self.len())
    }
}