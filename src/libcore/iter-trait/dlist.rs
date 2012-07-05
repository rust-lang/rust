type IMPL_T<A> = dlist::dlist<A>;

/**
 * Iterates through the current contents.
 *
 * Attempts to access this dlist during iteration are allowed (to allow for
 * e.g. breadth-first search with in-place enqueues), but removing the current
 * node is forbidden.
 */
fn EACH<A>(self: IMPL_T<A>, f: fn(A) -> bool) {
    import dlist::extensions;

    let mut link = self.peek_n();
    while option::is_some(link) {
        let nobe = option::get(link);
        // Check dlist invariant.
        if !option::is_some(nobe.root) ||
           !box::ptr_eq(*option::get(nobe.root), *self) {
            fail "Iteration encountered a dlist node not on this dlist."
        }
        f(nobe.data);
        // Check that the user didn't do a remove.
        // Note that this makes it ok for the user to remove the node and then
        // immediately put it back in a different position. I allow this.
        if !option::is_some(nobe.root) ||
           !box::ptr_eq(*option::get(nobe.root), *self) {
            fail "Removing a dlist node during iteration is forbidden!"
        }
        link = nobe.next_link();
    }
}

fn SIZE_HINT<A>(self: IMPL_T<A>) -> option<uint> {
    import dlist::extensions;
    some(self.len())
}
