type IMPL_T<A> = dlist::dlist<A>;

/**
 * Iterates through the current contents.
 *
 * Attempts to access this dlist during iteration are allowed (to allow for
 * e.g. breadth-first search with in-place enqueues), but removing the current
 * node is forbidden.
 */
pure fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    let mut link = self.peek_n();
    while option::is_some(link) {
        let nobe = option::get(link);
        assert nobe.linked;
        if !f(nobe.data) { break; }
        // Check (weakly) that the user didn't do a remove.
        if self.size == 0 {
            fail ~"The dlist became empty during iteration??"
        }
        if !nobe.linked ||
           (!((nobe.prev.is_some()
               || box::ptr_eq(*self.hd.expect(~"headless dlist?"), *nobe)) &&
              (nobe.next.is_some()
               || box::ptr_eq(*self.tl.expect(~"tailless dlist?"), *nobe)))) {
            fail ~"Removing a dlist node during iteration is forbidden!"
        }
        link = nobe.next_link();
    }
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    some(self.len())
}
