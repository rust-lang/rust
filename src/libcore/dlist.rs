/**
 * A doubly-linked list. Supports O(1) head, tail, count, push, pop, etc.
 *
 * Do not use ==, !=, <, etc on doubly-linked lists -- it may not terminate.
 */

import dlist_iter::extensions;

export dlist, dlist_node;
export new_dlist, from_elem, from_vec, extensions;

type dlist_link<T> = option<dlist_node<T>>;

enum dlist_node<T> = @{
    data: T,
    mut linked: bool, // for assertions
    mut prev: dlist_link<T>,
    mut next: dlist_link<T>
};

enum dlist<T> = @{
    mut size: uint,
    mut hd:   dlist_link<T>,
    mut tl:   dlist_link<T>,
};

impl private_methods<T> for dlist_node<T> {
    pure fn assert_links() {
        match self.next {
            some(neighbour) => match neighbour.prev {
              some(me) => if !box::ptr_eq(*self, *me) {
                  fail ~"Asymmetric next-link in dlist node."
              }
              none => fail ~"One-way next-link in dlist node."
            }
            none => ()
        }
        match self.prev {
            some(neighbour) => match neighbour.next {
              some(me) => if !box::ptr_eq(*me, *self) {
                  fail ~"Asymmetric prev-link in dlist node."
              }
              none => fail ~"One-way prev-link in dlist node."
            }
            none => ()
        }
    }
}

impl extensions<T> for dlist_node<T> {
    /// Get the next node in the list, if there is one.
    pure fn next_link() -> option<dlist_node<T>> {
        self.assert_links();
        self.next
    }
    /// Get the next node in the list, failing if there isn't one.
    pure fn next_node() -> dlist_node<T> {
        match self.next_link() {
            some(nobe) => nobe,
            none       => fail ~"This dlist node has no next neighbour."
        }
    }
    /// Get the previous node in the list, if there is one.
    pure fn prev_link() -> option<dlist_node<T>> {
        self.assert_links();
        self.prev
    }
    /// Get the previous node in the list, failing if there isn't one.
    pure fn prev_node() -> dlist_node<T> {
        match self.prev_link() {
            some(nobe) => nobe,
            none       => fail ~"This dlist node has no previous neighbour."
        }
    }
}

/// Creates a new dlist node with the given data.
pure fn new_dlist_node<T>(+data: T) -> dlist_node<T> {
    dlist_node(@{data: data, mut linked: false,
                 mut prev: none, mut next: none})
}

/// Creates a new, empty dlist.
pure fn new_dlist<T>() -> dlist<T> {
    dlist(@{mut size: 0, mut hd: none, mut tl: none})
}

/// Creates a new dlist with a single element
pure fn from_elem<T>(+data: T) -> dlist<T> {
    let list = new_dlist();
    unchecked { list.push(data); }
    list
}

fn from_vec<T: copy>(+vec: &[T]) -> dlist<T> {
    do vec::foldl(new_dlist(), vec) |list,data| {
        list.push(data); // Iterating left-to-right -- add newly to the tail.
        list
    }
}

/// Produce a list from a list of lists, leaving no elements behind in the
/// input. O(number of sub-lists).
fn concat<T>(lists: dlist<dlist<T>>) -> dlist<T> {
    let result = new_dlist();
    while !lists.is_empty() {
        result.append(lists.pop().get());
    }
    result
}

impl private_methods<T> for dlist<T> {
    pure fn new_link(-data: T) -> dlist_link<T> {
        some(dlist_node(@{data: data, mut linked: true,
                          mut prev: none, mut next: none}))
    }
    pure fn assert_mine(nobe: dlist_node<T>) {
        // These asserts could be stronger if we had node-root back-pointers,
        // but those wouldn't allow for O(1) append.
        if self.size == 0 {
            fail ~"This dlist is empty; that node can't be on it."
        }
        if !nobe.linked { fail ~"That node isn't linked to any dlist." }
        if !((nobe.prev.is_some()
              || box::ptr_eq(*self.hd.expect(~"headless dlist?"), *nobe)) &&
             (nobe.next.is_some()
              || box::ptr_eq(*self.tl.expect(~"tailless dlist?"), *nobe))) {
            fail ~"That node isn't on this dlist."
        }
    }
    fn make_mine(nobe: dlist_node<T>) {
        if nobe.prev.is_some() || nobe.next.is_some() || nobe.linked {
            fail ~"Cannot insert node that's already on a dlist!"
        }
        nobe.linked = true;
    }
    // Link two nodes together. If either of them are 'none', also sets
    // the head and/or tail pointers appropriately.
    #[inline(always)]
    fn link(+before: dlist_link<T>, +after: dlist_link<T>) {
        match before {
            some(neighbour) => neighbour.next = after,
            none            => self.hd        = after
        }
        match after {
            some(neighbour) => neighbour.prev = before,
            none            => self.tl        = before
        }
    }
    // Remove a node from the list.
    fn unlink(nobe: dlist_node<T>) {
        self.assert_mine(nobe);
        assert self.size > 0;
        self.link(nobe.prev, nobe.next);
        nobe.prev = none; // Release extraneous references.
        nobe.next = none;
        nobe.linked = false;
        self.size -= 1;
    }

    fn add_head(+nobe: dlist_link<T>) {
        self.link(nobe, self.hd); // Might set tail too.
        self.hd = nobe;
        self.size += 1;
    }
    fn add_tail(+nobe: dlist_link<T>) {
        self.link(self.tl, nobe); // Might set head too.
        self.tl = nobe;
        self.size += 1;
    }
    fn insert_left(nobe: dlist_link<T>, neighbour: dlist_node<T>) {
        self.assert_mine(neighbour);
        assert self.size > 0;
        self.link(neighbour.prev, nobe);
        self.link(nobe, some(neighbour));
        self.size += 1;
    }
    fn insert_right(neighbour: dlist_node<T>, nobe: dlist_link<T>) {
        self.assert_mine(neighbour);
        assert self.size > 0;
        self.link(nobe, neighbour.next);
        self.link(some(neighbour), nobe);
        self.size += 1;
    }
}

impl extensions<T> for dlist<T> {
    /// Get the size of the list. O(1).
    pure fn len()          -> uint { self.size }
    /// Returns true if the list is empty. O(1).
    pure fn is_empty()     -> bool { self.len() == 0 }
    /// Returns true if the list is not empty. O(1).
    pure fn is_not_empty() -> bool { self.len() != 0 }

    /// Add data to the head of the list. O(1).
    fn push_head(+data: T) {
        self.add_head(self.new_link(data));
    }
    /**
     * Add data to the head of the list, and get the new containing
     * node. O(1).
     */
    fn push_head_n(+data: T) -> dlist_node<T> {
        let mut nobe = self.new_link(data);
        self.add_head(nobe);
        option::get(nobe)
    }
    /// Add data to the tail of the list. O(1).
    fn push(+data: T) {
        self.add_tail(self.new_link(data));
    }
    /**
     * Add data to the tail of the list, and get the new containing
     * node. O(1).
     */
    fn push_n(+data: T) -> dlist_node<T> {
        let mut nobe = self.new_link(data);
        self.add_tail(nobe);
        option::get(nobe)
    }
    /**
     * Insert data into the middle of the list, left of the given node.
     * O(1).
     */
    fn insert_before(+data: T, neighbour: dlist_node<T>) {
        self.insert_left(self.new_link(data), neighbour);
    }
    /**
     * Insert an existing node in the middle of the list, left of the
     * given node. O(1).
     */
    fn insert_n_before(nobe: dlist_node<T>, neighbour: dlist_node<T>) {
        self.make_mine(nobe);
        self.insert_left(some(nobe), neighbour);
    }
    /**
     * Insert data in the middle of the list, left of the given node,
     * and get its containing node. O(1).
     */
    fn insert_before_n(+data: T, neighbour: dlist_node<T>) -> dlist_node<T> {
        let mut nobe = self.new_link(data);
        self.insert_left(nobe, neighbour);
        option::get(nobe)
    }
    /**
     * Insert data into the middle of the list, right of the given node.
     * O(1).
     */
    fn insert_after(+data: T, neighbour: dlist_node<T>) {
        self.insert_right(neighbour, self.new_link(data));
    }
    /**
     * Insert an existing node in the middle of the list, right of the
     * given node. O(1).
     */
    fn insert_n_after(nobe: dlist_node<T>, neighbour: dlist_node<T>) {
        self.make_mine(nobe);
        self.insert_right(neighbour, some(nobe));
    }
    /**
     * Insert data in the middle of the list, right of the given node,
     * and get its containing node. O(1).
     */
    fn insert_after_n(+data: T, neighbour: dlist_node<T>) -> dlist_node<T> {
        let mut nobe = self.new_link(data);
        self.insert_right(neighbour, nobe);
        option::get(nobe)
    }

    /// Remove a node from the head of the list. O(1).
    fn pop_n() -> option<dlist_node<T>> {
        let hd = self.peek_n();
        hd.map(|nobe| self.unlink(nobe));
        hd
    }
    /// Remove a node from the tail of the list. O(1).
    fn pop_tail_n() -> option<dlist_node<T>> {
        let tl = self.peek_tail_n();
        tl.map(|nobe| self.unlink(nobe));
        tl
    }
    /// Get the node at the list's head. O(1).
    pure fn peek_n() -> option<dlist_node<T>> { self.hd }
    /// Get the node at the list's tail. O(1).
    pure fn peek_tail_n() -> option<dlist_node<T>> { self.tl }

    /// Get the node at the list's head, failing if empty. O(1).
    pure fn head_n() -> dlist_node<T> {
        match self.hd {
            some(nobe) => nobe,
            none       => fail ~"Attempted to get the head of an empty dlist."
        }
    }
    /// Get the node at the list's tail, failing if empty. O(1).
    pure fn tail_n() -> dlist_node<T> {
        match self.tl {
            some(nobe) => nobe,
            none       => fail ~"Attempted to get the tail of an empty dlist."
        }
    }

    /// Remove a node from anywhere in the list. O(1).
    fn remove(nobe: dlist_node<T>) { self.unlink(nobe); }

    /**
     * Empty another list onto the end of this list, joining this list's tail
     * to the other list's head. O(1).
     */
    fn append(them: dlist<T>) {
        if box::ptr_eq(*self, *them) {
            fail ~"Cannot append a dlist to itself!"
        }
        if them.len() > 0 {
            self.link(self.tl, them.hd);
            self.tl    = them.tl;
            self.size += them.size;
            them.size  = 0;
            them.hd    = none;
            them.tl    = none;
        }
    }
    /**
     * Empty another list onto the start of this list, joining the other
     * list's tail to this list's head. O(1).
     */
    fn prepend(them: dlist<T>) {
        if box::ptr_eq(*self, *them) {
            fail ~"Cannot prepend a dlist to itself!"
        }
        if them.len() > 0 {
            self.link(them.tl, self.hd);
            self.hd    = them.hd;
            self.size += them.size;
            them.size  = 0;
            them.hd    = none;
            them.tl    = none;
        }
    }

    /// Reverse the list's elements in place. O(n).
    fn reverse() {
        do option::while_some(self.hd) |nobe| {
            let next_nobe = nobe.next;
            self.remove(nobe);
            self.make_mine(nobe);
            self.add_head(some(nobe));
            next_nobe
        }
    }

    /**
     * Remove everything from the list. This is important because the cyclic
     * links won't otherwise be automatically refcounted-collected. O(n).
     */
    fn clear() {
        // Cute as it would be to simply detach the list and proclaim "O(1)!",
        // the GC would still be a hidden O(n). Better to be honest about it.
        while !self.is_empty() {
            let _ = self.pop_n();
        }
    }

    /// Iterate over nodes.
    pure fn each_node(f: fn(dlist_node<T>) -> bool) {
        let mut link = self.peek_n();
        while link.is_some() {
            let nobe = link.get();
            if !f(nobe) { break; }
            link = nobe.next_link();
        }
    }

    /// Check data structure integrity. O(n).
    fn assert_consistent() {
        if option::is_none(self.hd) || option::is_none(self.tl) {
            assert option::is_none(self.hd) && option::is_none(self.tl);
        }
        // iterate forwards
        let mut count = 0;
        let mut link = self.peek_n();
        let mut rabbit = link;
        while option::is_some(link) {
            let nobe = option::get(link);
            assert nobe.linked;
            // check cycle
            if option::is_some(rabbit) { rabbit = option::get(rabbit).next; }
            if option::is_some(rabbit) { rabbit = option::get(rabbit).next; }
            if option::is_some(rabbit) {
                assert !box::ptr_eq(*option::get(rabbit), *nobe);
            }
            // advance
            link = nobe.next_link();
            count += 1;
        }
        assert count == self.len();
        // iterate backwards - some of this is probably redundant.
        link = self.peek_tail_n();
        rabbit = link;
        while option::is_some(link) {
            let nobe = option::get(link);
            assert nobe.linked;
            // check cycle
            if option::is_some(rabbit) { rabbit = option::get(rabbit).prev; }
            if option::is_some(rabbit) { rabbit = option::get(rabbit).prev; }
            if option::is_some(rabbit) {
                assert !box::ptr_eq(*option::get(rabbit), *nobe);
            }
            // advance
            link = nobe.prev_link();
            count -= 1;
        }
        assert count == 0;
    }
}

impl extensions<T: copy> for dlist<T> {
    /// Remove data from the head of the list. O(1).
    fn pop()       -> option<T> { self.pop_n().map       (|nobe| nobe.data) }
    /// Remove data from the tail of the list. O(1).
    fn pop_tail()  -> option<T> { self.pop_tail_n().map  (|nobe| nobe.data) }
    /// Get data at the list's head. O(1).
    pure fn peek() -> option<T> { self.peek_n().map      (|nobe| nobe.data) }
    /// Get data at the list's tail. O(1).
    pure fn peek_tail() -> option<T> {
        self.peek_tail_n().map (|nobe| nobe.data)
    }
    /// Get data at the list's head, failing if empty. O(1).
    pure fn head() -> T { self.head_n().data }
    /// Get data at the list's tail, failing if empty. O(1).
    pure fn tail() -> T { self.tail_n().data }
    /// Get the elements of the list as a vector. O(n).
    pure fn to_vec() -> ~[mut T] {
        let mut v = ~[mut];
        unchecked {
            vec::reserve(v, self.size);
            // Take this out of the unchecked when iter's functions are pure
            for self.eachi |index,data| {
                v[index] = data;
            }
        }
        v
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_dlist_concat() {
        let a = from_vec(~[1,2]);
        let b = from_vec(~[3,4]);
        let c = from_vec(~[5,6]);
        let d = from_vec(~[7,8]);
        let ab = from_vec(~[a,b]);
        let cd = from_vec(~[c,d]);
        let abcd = concat(concat(from_vec(~[ab,cd])));
        abcd.assert_consistent(); assert abcd.len() == 8;
        abcd.assert_consistent(); assert abcd.pop().get() == 1;
        abcd.assert_consistent(); assert abcd.pop().get() == 2;
        abcd.assert_consistent(); assert abcd.pop().get() == 3;
        abcd.assert_consistent(); assert abcd.pop().get() == 4;
        abcd.assert_consistent(); assert abcd.pop().get() == 5;
        abcd.assert_consistent(); assert abcd.pop().get() == 6;
        abcd.assert_consistent(); assert abcd.pop().get() == 7;
        abcd.assert_consistent(); assert abcd.pop().get() == 8;
        abcd.assert_consistent(); assert abcd.is_empty();
    }
    #[test]
    fn test_dlist_append() {
        let a = from_vec(~[1,2,3]);
        let b = from_vec(~[4,5,6]);
        a.append(b);
        assert a.len() == 6;
        assert b.len() == 0;
        b.assert_consistent();
        a.assert_consistent(); assert a.pop().get() == 1;
        a.assert_consistent(); assert a.pop().get() == 2;
        a.assert_consistent(); assert a.pop().get() == 3;
        a.assert_consistent(); assert a.pop().get() == 4;
        a.assert_consistent(); assert a.pop().get() == 5;
        a.assert_consistent(); assert a.pop().get() == 6;
        a.assert_consistent(); assert a.is_empty();
    }
    #[test]
    fn test_dlist_append_empty() {
        let a = from_vec(~[1,2,3]);
        let b = new_dlist::<int>();
        a.append(b);
        assert a.len() == 3;
        assert b.len() == 0;
        b.assert_consistent();
        a.assert_consistent(); assert a.pop().get() == 1;
        a.assert_consistent(); assert a.pop().get() == 2;
        a.assert_consistent(); assert a.pop().get() == 3;
        a.assert_consistent(); assert a.is_empty();
    }
    #[test]
    fn test_dlist_append_to_empty() {
        let a = new_dlist::<int>();
        let b = from_vec(~[4,5,6]);
        a.append(b);
        assert a.len() == 3;
        assert b.len() == 0;
        b.assert_consistent();
        a.assert_consistent(); assert a.pop().get() == 4;
        a.assert_consistent(); assert a.pop().get() == 5;
        a.assert_consistent(); assert a.pop().get() == 6;
        a.assert_consistent(); assert a.is_empty();
    }
    #[test]
    fn test_dlist_append_two_empty() {
        let a = new_dlist::<int>();
        let b = new_dlist::<int>();
        a.append(b);
        assert a.len() == 0;
        assert b.len() == 0;
        b.assert_consistent();
        a.assert_consistent();
    }
    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_dlist_append_self() {
        let a = new_dlist::<int>();
        a.append(a);
    }
    #[test]
    #[ignore(cfg(windows))]
    #[should_fail]
    fn test_dlist_prepend_self() {
        let a = new_dlist::<int>();
        a.prepend(a);
    }
    #[test]
    fn test_dlist_prepend() {
        let a = from_vec(~[1,2,3]);
        let b = from_vec(~[4,5,6]);
        b.prepend(a);
        assert a.len() == 0;
        assert b.len() == 6;
        a.assert_consistent();
        b.assert_consistent(); assert b.pop().get() == 1;
        b.assert_consistent(); assert b.pop().get() == 2;
        b.assert_consistent(); assert b.pop().get() == 3;
        b.assert_consistent(); assert b.pop().get() == 4;
        b.assert_consistent(); assert b.pop().get() == 5;
        b.assert_consistent(); assert b.pop().get() == 6;
        b.assert_consistent(); assert b.is_empty();
    }
    #[test]
    fn test_dlist_reverse() {
        let a = from_vec(~[5,4,3,2,1]);
        a.reverse();
        assert a.len() == 5;
        a.assert_consistent(); assert a.pop().get() == 1;
        a.assert_consistent(); assert a.pop().get() == 2;
        a.assert_consistent(); assert a.pop().get() == 3;
        a.assert_consistent(); assert a.pop().get() == 4;
        a.assert_consistent(); assert a.pop().get() == 5;
        a.assert_consistent(); assert a.is_empty();
    }
    #[test]
    fn test_dlist_reverse_empty() {
        let a = new_dlist::<int>();
        a.reverse();
        assert a.len() == 0;
        a.assert_consistent();
    }
    #[test]
    fn test_dlist_each_node() {
        let a = from_vec(~[1,2,4,5]);
        for a.each_node |nobe| {
            if nobe.data > 3 {
                a.insert_before(3, nobe);
            }
        }
        assert a.len() == 6;
        a.assert_consistent(); assert a.pop().get() == 1;
        a.assert_consistent(); assert a.pop().get() == 2;
        a.assert_consistent(); assert a.pop().get() == 3;
        a.assert_consistent(); assert a.pop().get() == 4;
        a.assert_consistent(); assert a.pop().get() == 3;
        a.assert_consistent(); assert a.pop().get() == 5;
        a.assert_consistent(); assert a.is_empty();
    }
    #[test]
    fn test_dlist_clear() {
        let a = from_vec(~[5,4,3,2,1]);
        a.clear();
        assert a.len() == 0;
        a.assert_consistent();
    }
    #[test]
    fn test_dlist_is_empty() {
        let empty = new_dlist::<int>();
        let full1 = from_vec(~[1,2,3]);
        assert empty.is_empty();
        assert !full1.is_empty();
        assert !empty.is_not_empty();
        assert full1.is_not_empty();
    }
    #[test]
    fn test_dlist_head_tail() {
        let l = from_vec(~[1,2,3]);
        assert l.head() == 1;
        assert l.tail() == 3;
        assert l.len() == 3;
    }
    #[test]
    fn test_dlist_pop() {
        let l = from_vec(~[1,2,3]);
        assert l.pop().get() == 1;
        assert l.tail() == 3;
        assert l.head() == 2;
        assert l.pop().get() == 2;
        assert l.tail() == 3;
        assert l.head() == 3;
        assert l.pop().get() == 3;
        assert l.is_empty();
        assert l.pop().is_none();
    }
    #[test]
    fn test_dlist_pop_tail() {
        let l = from_vec(~[1,2,3]);
        assert l.pop_tail().get() == 3;
        assert l.tail() == 2;
        assert l.head() == 1;
        assert l.pop_tail().get() == 2;
        assert l.tail() == 1;
        assert l.head() == 1;
        assert l.pop_tail().get() == 1;
        assert l.is_empty();
        assert l.pop_tail().is_none();
    }
    #[test]
    fn test_dlist_push() {
        let l = new_dlist::<int>();
        l.push(1);
        assert l.head() == 1;
        assert l.tail() == 1;
        l.push(2);
        assert l.head() == 1;
        assert l.tail() == 2;
        l.push(3);
        assert l.head() == 1;
        assert l.tail() == 3;
        assert l.len() == 3;
    }
    #[test]
    fn test_dlist_push_head() {
        let l = new_dlist::<int>();
        l.push_head(3);
        assert l.head() == 3;
        assert l.tail() == 3;
        l.push_head(2);
        assert l.head() == 2;
        assert l.tail() == 3;
        l.push_head(1);
        assert l.head() == 1;
        assert l.tail() == 3;
        assert l.len() == 3;
    }
    #[test]
    fn test_dlist_foldl() {
        let l = from_vec(vec::from_fn(101, |x|x));
        assert iter::foldl(l, 0, |accum,elem| accum+elem) == 5050;
    }
    #[test]
    fn test_dlist_break_early() {
        let l = from_vec(~[1,2,3,4,5]);
        let mut x = 0;
        for l.each |i| {
            x += 1;
            if (i == 3) { break; }
        }
        assert x == 3;
    }
    #[test]
    fn test_dlist_remove_head() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let _two = l.push_n(2);
        l.assert_consistent(); let _three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(one);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); assert l.head() == 2;
        l.assert_consistent(); assert l.tail() == 3;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_mid() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let _one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); let _three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(two);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 3;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_tail() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let _one = l.push_n(1);
        l.assert_consistent(); let _two = l.push_n(2);
        l.assert_consistent(); let three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(three);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 2;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_one_two() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); let _three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(one);
        l.assert_consistent(); l.remove(two);
        // and through and through, the vorpal blade went snicker-snack
        l.assert_consistent(); assert l.len() == 1;
        l.assert_consistent(); assert l.head() == 3;
        l.assert_consistent(); assert l.tail() == 3;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_one_three() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let _two = l.push_n(2);
        l.assert_consistent(); let three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(one);
        l.assert_consistent(); l.remove(three);
        l.assert_consistent(); assert l.len() == 1;
        l.assert_consistent(); assert l.head() == 2;
        l.assert_consistent(); assert l.tail() == 2;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_two_three() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let _one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); let three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(two);
        l.assert_consistent(); l.remove(three);
        l.assert_consistent(); assert l.len() == 1;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 1;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_remove_all() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); let three = l.push_n(3);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); l.remove(two);
        l.assert_consistent(); l.remove(three);
        l.assert_consistent(); l.remove(one); // Twenty-three is number one!
        l.assert_consistent(); assert l.peek() == none;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_insert_n_before() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let _one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); let three = new_dlist_node(3);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); l.insert_n_before(three, two);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 2;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_insert_n_after() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let _two = l.push_n(2);
        l.assert_consistent(); let three = new_dlist_node(3);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); l.insert_n_after(three, one);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 2;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_insert_before_head() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let one = l.push_n(1);
        l.assert_consistent(); let _two = l.push_n(2);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); l.insert_before(3, one);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); assert l.head() == 3;
        l.assert_consistent(); assert l.tail() == 2;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test]
    fn test_dlist_insert_after_tail() {
        let l = new_dlist::<int>();
        l.assert_consistent(); let _one = l.push_n(1);
        l.assert_consistent(); let two = l.push_n(2);
        l.assert_consistent(); assert l.len() == 2;
        l.assert_consistent(); l.insert_after(3, two);
        l.assert_consistent(); assert l.len() == 3;
        l.assert_consistent(); assert l.head() == 1;
        l.assert_consistent(); assert l.tail() == 3;
        l.assert_consistent(); assert l.pop().get() == 1;
        l.assert_consistent(); assert l.pop().get() == 2;
        l.assert_consistent(); assert l.pop().get() == 3;
        l.assert_consistent(); assert l.is_empty();
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_asymmetric_link() {
        let l = new_dlist::<int>();
        let _one = l.push_n(1);
        let two = l.push_n(2);
        two.prev = none;
        l.assert_consistent();
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_cyclic_list() {
        let l = new_dlist::<int>();
        let one = l.push_n(1);
        let _two = l.push_n(2);
        let three = l.push_n(3);
        three.next = some(one);
        one.prev = some(three);
        l.assert_consistent();
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_headless() {
        new_dlist::<int>().head();
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_insert_already_present_before() {
        let l = new_dlist::<int>();
        let one = l.push_n(1);
        let two = l.push_n(2);
        l.insert_n_before(two, one);
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_insert_already_present_after() {
        let l = new_dlist::<int>();
        let one = l.push_n(1);
        let two = l.push_n(2);
        l.insert_n_after(one, two);
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_insert_before_orphan() {
        let l = new_dlist::<int>();
        let one = new_dlist_node(1);
        let two = new_dlist_node(2);
        l.insert_n_before(one, two);
    }
    #[test] #[should_fail] #[ignore(cfg(windows))]
    fn test_dlist_insert_after_orphan() {
        let l = new_dlist::<int>();
        let one = new_dlist_node(1);
        let two = new_dlist_node(2);
        l.insert_n_after(two, one);
    }
}
