// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A doubly-linked list with owned nodes.
//!
//! The DList allows pushing and popping elements at either end.
//!
//! DList implements the trait Deque. It should be imported with `use
//! extra::container::Deque`.


// DList is constructed like a singly-linked list over the field `next`.
// including the last link being None; each Node owns its `next` field.
//
// Backlinks over DList::prev are raw pointers that form a full chain in
// the reverse direction.

use std::cast;
use std::cmp;
use std::ptr;
use std::util;
use std::iterator::{FromIterator, InvertIterator};

use container::Deque;

/// A doubly-linked list.
pub struct DList<T> {
    priv length: uint,
    priv list_head: Link<T>,
    priv list_tail: Rawlink<Node<T>>,
}

type Link<T> = Option<~Node<T>>;
struct Rawlink<T> { priv p: *mut T }

struct Node<T> {
    priv next: Link<T>,
    priv prev: Rawlink<Node<T>>,
    priv value: T,
}

/// Double-ended DList iterator
pub struct DListIterator<'self, T> {
    priv head: &'self Link<T>,
    priv tail: Rawlink<Node<T>>,
    priv nelem: uint,
}

/// Double-ended mutable DList iterator
pub struct MutDListIterator<'self, T> {
    priv list: &'self mut DList<T>,
    priv head: Rawlink<Node<T>>,
    priv tail: Rawlink<Node<T>>,
    priv nelem: uint,
}

/// DList consuming iterator
pub struct ConsumeIterator<T> {
    priv list: DList<T>
}

/// Rawlink is a type like Option<T> but for holding a raw pointer
impl<T> Rawlink<T> {
    /// Like Option::None for Rawlink
    fn none() -> Rawlink<T> {
        Rawlink{p: ptr::mut_null()}
    }

    /// Like Option::Some for Rawlink
    fn some(n: &mut T) -> Rawlink<T> {
        Rawlink{p: ptr::to_mut_unsafe_ptr(n)}
    }

    /// Convert the `Rawlink` into an Option value
    fn resolve_immut(&self) -> Option<&T> {
        unsafe { self.p.to_option() }
    }

    /// Convert the `Rawlink` into an Option value
    fn resolve(&mut self) -> Option<&mut T> {
        if self.p.is_null() {
            None
        } else {
            Some(unsafe { cast::transmute(self.p) })
        }
    }
}

/// Set the .prev field on `next`, then return `Some(next)`
fn link_with_prev<T>(mut next: ~Node<T>, prev: Rawlink<Node<T>>) -> Link<T> {
    next.prev = prev;
    Some(next)
}

impl<T> Container for DList<T> {
    /// O(1)
    fn is_empty(&self) -> bool {
        self.list_head.is_none()
    }
    /// O(1)
    fn len(&self) -> uint {
        self.length
    }
}

impl<T> Mutable for DList<T> {
    /// Remove all elements from the DList
    ///
    /// O(N)
    fn clear(&mut self) {
        *self = DList::new()
    }
}

impl<T> Deque<T> for DList<T> {
    /// Provide a reference to the front element, or None if the list is empty
    fn front<'a>(&'a self) -> Option<&'a T> {
        self.list_head.chain_ref(|x| Some(&x.value))
    }

    /// Provide a mutable reference to the front element, or None if the list is empty
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        match self.list_head {
            None => None,
            Some(ref mut head) => Some(&mut head.value),
        }
    }

    /// Provide a reference to the back element, or None if the list is empty
    fn back<'a>(&'a self) -> Option<&'a T> {
        match self.list_tail.resolve_immut() {
            None => None,
            Some(tail) => Some(&tail.value),
        }
    }

    /// Provide a mutable reference to the back element, or None if the list is empty
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T> {
        match self.list_tail.resolve() {
            None => None,
            Some(tail) => Some(&mut tail.value),
        }
    }

    /// Add an element last in the list
    ///
    /// O(1)
    fn push_back(&mut self, elt: T) {
        match self.list_tail.resolve() {
            None => return self.push_front(elt),
            Some(tail) => {
                let mut new_tail = ~Node{value: elt, next: None, prev: self.list_tail};
                self.list_tail = Rawlink::some(new_tail);
                tail.next = Some(new_tail);
            }
        }
        self.length += 1;
    }

    /// Remove the last element and return it, or None if the list is empty
    ///
    /// O(1)
    #[inline]
    fn pop_back(&mut self) -> Option<T> {
        match self.list_tail.resolve() {
            None => None,
            Some(tail) => {
                self.length -= 1;
                let tail_own = match tail.prev.resolve() {
                    None => {
                        self.list_tail = Rawlink::none();
                        self.list_head.swap_unwrap()
                    },
                    Some(tail_prev) => {
                        self.list_tail = tail.prev;
                        tail_prev.next.swap_unwrap()
                    }
                };
                Some(tail_own.value)
            }
        }
    }

    /// Add an element first in the list
    ///
    /// O(1)
    fn push_front(&mut self, elt: T) {
        let mut new_head = ~Node{value: elt, next: None, prev: Rawlink::none()};
        match self.list_head {
            None => {
                self.list_tail = Rawlink::some(new_head);
                self.list_head = Some(new_head);
            }
            Some(ref mut head) => {
                head.prev = Rawlink::some(new_head);
                util::swap(head, &mut new_head);
                head.next = Some(new_head);
            }
        }
        self.length += 1;
    }

    /// Remove the first element and return it, or None if the list is empty
    ///
    /// O(1)
    fn pop_front(&mut self) -> Option<T> {
        match util::replace(&mut self.list_head, None) {
            None => None,
            Some(old_head) => {
                self.length -= 1;
                match *old_head {
                    Node{value: value, next: Some(next), prev: _} => {
                        self.list_head = link_with_prev(next, Rawlink::none());
                        Some(value)
                    }
                    Node{value: value, next: None, prev: _} => {
                        self.list_tail = Rawlink::none();
                        Some(value)
                    }
                }
            }
        }
    }
}

impl<T> DList<T> {
    /// Create an empty DList
    #[inline]
    pub fn new() -> DList<T> {
        DList{list_head: None, list_tail: Rawlink::none(), length: 0}
    }

    /// Add all elements from `other` to the end of the list
    ///
    /// O(1)
    pub fn append(&mut self, other: DList<T>) {
        match self.list_tail.resolve() {
            None => *self = other,
            Some(tail) => {
                match other {
                    DList{list_head: None, list_tail: _, length: _} => return,
                    DList{list_head: Some(node), list_tail: o_tail, length: o_length} => {
                        tail.next = link_with_prev(node, self.list_tail);
                        self.list_tail = o_tail;
                        self.length += o_length;
                    }
                }
            }
        }
    }

    /// Add all elements from `other` to the beginning of the list
    ///
    /// O(1)
    pub fn prepend(&mut self, mut other: DList<T>) {
        util::swap(self, &mut other);
        self.append(other);
    }

    /// Insert `elt` before the first `x` in the list where `f(x, elt)` is true,
    /// or at the end.
    ///
    /// O(N)
    #[inline]
    pub fn insert_when(&mut self, elt: T, f: &fn(&T, &T) -> bool) {
        {
            let mut it = self.mut_iter();
            loop {
                match it.peek_next() {
                    None => break,
                    Some(x) => if f(x, &elt) { break }
                }
                it.next();
            }
            it.insert_next(elt);
        }
    }

    /// Merge DList `other` into this DList, using the function `f`.
    /// Iterate the both DList with `a` from self and `b` from `other`, and
    /// put `a` in the result if `f(a, b)` is true, else `b`.
    ///
    /// O(max(N, M))
    pub fn merge(&mut self, mut other: DList<T>, f: &fn(&T, &T) -> bool) {
        {
            let mut it = self.mut_iter();
            loop {
                let take_a = match (it.peek_next(), other.front()) {
                    (_   , None) => return,
                    (None, _   ) => break,
                    (Some(ref mut x), Some(y)) => f(*x, y),
                };
                if take_a {
                    it.next();
                } else {
                    it.insert_next(other.pop_front().unwrap());
                }
            }
        }
        self.append(other);
    }


    /// Provide a forward iterator
    pub fn iter<'a>(&'a self) -> DListIterator<'a, T> {
        DListIterator{nelem: self.len(), head: &self.list_head, tail: self.list_tail}
    }

    /// Provide a reverse iterator
    pub fn rev_iter<'a>(&'a self) -> InvertIterator<&'a T, DListIterator<'a, T>> {
        self.iter().invert()
    }

    /// Provide a forward iterator with mutable references
    pub fn mut_iter<'a>(&'a mut self) -> MutDListIterator<'a, T> {
        let head_raw = match self.list_head {
            Some(ref mut h) => Rawlink::some(*h),
            None => Rawlink::none(),
        };
        MutDListIterator{
            nelem: self.len(),
            head: head_raw,
            tail: self.list_tail,
            list: self
        }
    }
    /// Provide a reverse iterator with mutable references
    pub fn mut_rev_iter<'a>(&'a mut self) -> InvertIterator<&'a mut T,
                                                MutDListIterator<'a, T>> {
        self.mut_iter().invert()
    }


    /// Consume the list into an iterator yielding elements by value
    pub fn consume_iter(self) -> ConsumeIterator<T> {
        ConsumeIterator{list: self}
    }

    /// Consume the list into an iterator yielding elements by value, in reverse
    pub fn consume_rev_iter(self) -> InvertIterator<T, ConsumeIterator<T>> {
        self.consume_iter().invert()
    }
}

impl<T: cmp::TotalOrd> DList<T> {
    /// Insert `elt` sorted in ascending order
    ///
    /// O(N)
    pub fn insert_ordered(&mut self, elt: T) {
        self.insert_when(elt, |a, b| a.cmp(b) != cmp::Less);
    }
}

impl<'self, A> Iterator<&'self A> for DListIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self A> {
        if self.nelem == 0 {
            return None;
        }
        match *self.head {
            None => None,
            Some(ref head) => {
                self.nelem -= 1;
                self.head = &head.next;
                Some(&head.value)
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.nelem, Some(self.nelem))
    }
}

impl<'self, A> DoubleEndedIterator<&'self A> for DListIterator<'self, A> {
    fn next_back(&mut self) -> Option<&'self A> {
        if self.nelem == 0 {
            return None;
        }
        match self.tail.resolve() {
            None => None,
            Some(prev) => {
                self.nelem -= 1;
                self.tail = prev.prev;
                Some(&prev.value)
            }
        }
    }
}

impl<'self, A> Iterator<&'self mut A> for MutDListIterator<'self, A> {
    #[inline]
    fn next(&mut self) -> Option<&'self mut A> {
        if self.nelem == 0 {
            return None;
        }
        match self.head.resolve() {
            None => None,
            Some(next) => {
                self.nelem -= 1;
                self.head = match next.next {
                    Some(ref mut node) => Rawlink::some(&mut **node),
                    None => Rawlink::none(),
                };
                Some(&mut next.value)
            }
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.nelem, Some(self.nelem))
    }
}

impl<'self, A> DoubleEndedIterator<&'self mut A> for MutDListIterator<'self, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'self mut A> {
        if self.nelem == 0 {
            return None;
        }
        match self.tail.resolve() {
            None => None,
            Some(prev) => {
                self.nelem -= 1;
                self.tail = prev.prev;
                Some(&mut prev.value)
            }
        }
    }
}


/// Allow mutating the DList while iterating
pub trait ListInsertion<A> {
    /// Insert `elt` just after to the most recently yielded element
    fn insert_next(&mut self, elt: A);

    /// Provide a reference to the next element, without changing the iterator
    fn peek_next<'a>(&'a mut self) -> Option<&'a mut A>;
}

impl<'self, A> ListInsertion<A> for MutDListIterator<'self, A> {
    fn insert_next(&mut self, elt: A) {
        // Insert an element before `self.head` so that it is between the
        // previously yielded element and self.head.
        match self.head.resolve() {
            None => { self.list.push_back(elt); }
            Some(node) => {
                let prev_node = match node.prev.resolve() {
                    None => return self.list.push_front(elt),
                    Some(prev) => prev,
                };
                let mut ins_node = ~Node{value: elt, next: None, prev: Rawlink::none()};
                let node_own = prev_node.next.swap_unwrap();
                ins_node.next = link_with_prev(node_own, Rawlink::some(ins_node));
                prev_node.next = link_with_prev(ins_node, Rawlink::some(prev_node));
                self.list.length += 1;
            }
        }
    }

    fn peek_next<'a>(&'a mut self) -> Option<&'a mut A> {
        match self.head.resolve() {
            None => None,
            Some(head) => Some(&mut head.value),
        }
    }
}

impl<A> Iterator<A> for ConsumeIterator<A> {
    fn next(&mut self) -> Option<A> { self.list.pop_front() }
    fn size_hint(&self) -> (uint, Option<uint>) {
        (self.list.length, Some(self.list.length))
    }
}

impl<A> DoubleEndedIterator<A> for ConsumeIterator<A> {
    fn next_back(&mut self) -> Option<A> { self.list.pop_back() }
}

impl<A, T: Iterator<A>> FromIterator<A, T> for DList<A> {
    fn from_iterator(iterator: &mut T) -> DList<A> {
        let mut ret = DList::new();
        for iterator.advance |elt| { ret.push_back(elt); }
        ret
    }
}

impl<A: Eq> Eq for DList<A> {
    fn eq(&self, other: &DList<A>) -> bool {
        self.len() == other.len() &&
            self.iter().zip(other.iter()).all(|(a, b)| a.eq(b))
    }
    fn ne(&self, other: &DList<A>) -> bool {
        !self.eq(other)
    }
}

impl<A: Clone> Clone for DList<A> {
    fn clone(&self) -> DList<A> {
        self.iter().transform(|x| x.clone()).collect()
    }
}

#[cfg(test)]
pub fn check_links<T>(list: &DList<T>) {
    let mut len = 0u;
    let mut last_ptr: Option<&Node<T>> = None;
    let mut node_ptr: &Node<T>;
    match list.list_head {
        None => { assert_eq!(0u, list.length); return }
        Some(ref node) => node_ptr = &**node,
    }
    loop {
        match (last_ptr, node_ptr.prev.resolve_immut()) {
            (None   , None      ) => {}
            (None   , _         ) => fail!("prev link for list_head"),
            (Some(p), Some(pptr)) => {
                assert_eq!(p as *Node<T>, pptr as *Node<T>);
            }
            _ => fail!("prev link is none, not good"),
        }
        match node_ptr.next {
            Some(ref next) => {
                last_ptr = Some(node_ptr);
                node_ptr = &**next;
                len += 1;
            }
            None => {
                len += 1;
                break;
            }
        }
    }
    assert_eq!(len, list.length);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rand;
    use std::int;
    use extra::test;

    #[test]
    fn test_basic() {
        let mut m = DList::new::<~int>();
        assert_eq!(m.pop_front(), None);
        assert_eq!(m.pop_back(), None);
        assert_eq!(m.pop_front(), None);
        m.push_front(~1);
        assert_eq!(m.pop_front(), Some(~1));
        m.push_back(~2);
        m.push_back(~3);
        assert_eq!(m.len(), 2);
        assert_eq!(m.pop_front(), Some(~2));
        assert_eq!(m.pop_front(), Some(~3));
        assert_eq!(m.len(), 0);
        assert_eq!(m.pop_front(), None);
        m.push_back(~1);
        m.push_back(~3);
        m.push_back(~5);
        m.push_back(~7);
        assert_eq!(m.pop_front(), Some(~1));

        let mut n = DList::new();
        n.push_front(2);
        n.push_front(3);
        {
            assert_eq!(n.front().unwrap(), &3);
            let x = n.front_mut().unwrap();
            assert_eq!(*x, 3);
            *x = 0;
        }
        {
            assert_eq!(n.back().unwrap(), &2);
            let y = n.back_mut().unwrap();
            assert_eq!(*y, 2);
            *y = 1;
        }
        assert_eq!(n.pop_front(), Some(0));
        assert_eq!(n.pop_front(), Some(1));
    }

    #[cfg(test)]
    fn generate_test() -> DList<int> {
        list_from(&[0,1,2,3,4,5,6])
    }

    #[cfg(test)]
    fn list_from<T: Copy>(v: &[T]) -> DList<T> {
        v.iter().transform(|x| copy *x).collect()
    }

    #[test]
    fn test_append() {
        {
            let mut m = DList::new();
            let mut n = DList::new();
            n.push_back(2);
            m.append(n);
            assert_eq!(m.len(), 1);
            assert_eq!(m.pop_back(), Some(2));
            check_links(&m);
        }
        {
            let mut m = DList::new();
            let n = DList::new();
            m.push_back(2);
            m.append(n);
            assert_eq!(m.len(), 1);
            assert_eq!(m.pop_back(), Some(2));
            check_links(&m);
        }

        let v = ~[1,2,3,4,5];
        let u = ~[9,8,1,2,3,4,5];
        let mut m = list_from(v);
        m.append(list_from(u));
        check_links(&m);
        let sum = v + u;
        assert_eq!(sum.len(), m.len());
        for sum.consume_iter().advance |elt| {
            assert_eq!(m.pop_front(), Some(elt))
        }
    }

    #[test]
    fn test_prepend() {
        {
            let mut m = DList::new();
            let mut n = DList::new();
            n.push_back(2);
            m.prepend(n);
            assert_eq!(m.len(), 1);
            assert_eq!(m.pop_back(), Some(2));
            check_links(&m);
        }

        let v = ~[1,2,3,4,5];
        let u = ~[9,8,1,2,3,4,5];
        let mut m = list_from(v);
        m.prepend(list_from(u));
        check_links(&m);
        let sum = u + v;
        assert_eq!(sum.len(), m.len());
        for sum.consume_iter().advance |elt| {
            assert_eq!(m.pop_front(), Some(elt))
        }
    }

    #[test]
    fn test_iterator() {
        let m = generate_test();
        for m.iter().enumerate().advance |(i, elt)| {
            assert_eq!(i as int, *elt);
        }
        let mut n = DList::new();
        assert_eq!(n.iter().next(), None);
        n.push_front(4);
        let mut it = n.iter();
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &4);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_iterator_double_end() {
        let mut n = DList::new();
        assert_eq!(n.iter().next(), None);
        n.push_front(4);
        n.push_front(5);
        n.push_front(6);
        let mut it = n.iter();
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.next().unwrap(), &6);
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(it.next_back().unwrap(), &4);
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next_back().unwrap(), &5);
        assert_eq!(it.next_back(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_rev_iter() {
        let m = generate_test();
        for m.rev_iter().enumerate().advance |(i, elt)| {
            assert_eq!((6 - i) as int, *elt);
        }
        let mut n = DList::new();
        assert_eq!(n.rev_iter().next(), None);
        n.push_front(4);
        let mut it = n.rev_iter();
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &4);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_mut_iter() {
        let mut m = generate_test();
        let mut len = m.len();
        for m.mut_iter().enumerate().advance |(i, elt)| {
            assert_eq!(i as int, *elt);
            len -= 1;
        }
        assert_eq!(len, 0);
        let mut n = DList::new();
        assert!(n.mut_iter().next().is_none());
        n.push_front(4);
        n.push_back(5);
        let mut it = n.mut_iter();
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_iterator_mut_double_end() {
        let mut n = DList::new();
        assert!(n.mut_iter().next_back().is_none());
        n.push_front(4);
        n.push_front(5);
        n.push_front(6);
        let mut it = n.mut_iter();
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(*it.next().unwrap(), 6);
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert_eq!(*it.next_back().unwrap(), 4);
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(*it.next_back().unwrap(), 5);
        assert!(it.next_back().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn test_insert_prev() {
        let mut m = list_from(&[0,2,4,6,8]);
        let len = m.len();
        {
            let mut it = m.mut_iter();
            it.insert_next(-2);
            loop {
                match it.next() {
                    None => break,
                    Some(elt) => {
                        it.insert_next(*elt + 1);
                        match it.peek_next() {
                            Some(x) => assert_eq!(*x, *elt + 2),
                            None => assert_eq!(8, *elt),
                        }
                    }
                }
            }
            it.insert_next(0);
            it.insert_next(1);
        }
        check_links(&m);
        assert_eq!(m.len(), 3 + len * 2);
        assert_eq!(m.consume_iter().collect::<~[int]>(), ~[-2,0,1,2,3,4,5,6,7,8,9,0,1]);
    }

    #[test]
    fn test_merge() {
        let mut m = list_from([0, 1, 3, 5, 6, 7, 2]);
        let n = list_from([-1, 0, 0, 7, 7, 9]);
        let len = m.len() + n.len();
        m.merge(n, |a, b| a <= b);
        assert_eq!(m.len(), len);
        check_links(&m);
        let res = m.consume_iter().collect::<~[int]>();
        assert_eq!(res, ~[-1, 0, 0, 0, 1, 3, 5, 6, 7, 2, 7, 7, 9]);
    }

    #[test]
    fn test_insert_ordered() {
        let mut n = DList::new();
        n.insert_ordered(1);
        assert_eq!(n.len(), 1);
        assert_eq!(n.pop_front(), Some(1));

        let mut m = DList::new();
        m.push_back(2);
        m.push_back(4);
        m.insert_ordered(3);
        check_links(&m);
        assert_eq!(~[2,3,4], m.consume_iter().collect::<~[int]>());
    }

    #[test]
    fn test_mut_rev_iter() {
        let mut m = generate_test();
        for m.mut_rev_iter().enumerate().advance |(i, elt)| {
            assert_eq!((6-i) as int, *elt);
        }
        let mut n = DList::new();
        assert!(n.mut_rev_iter().next().is_none());
        n.push_front(4);
        let mut it = n.mut_rev_iter();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
    }

    #[test]
    fn test_send() {
        let n = list_from([1,2,3]);
        do spawn {
            check_links(&n);
            assert_eq!(~[&1,&2,&3], n.iter().collect::<~[&int]>());
        }
    }

    #[test]
    fn test_eq() {
        let mut n: DList<u8> = list_from([]);
        let mut m = list_from([]);
        assert_eq!(&n, &m);
        n.push_front(1);
        assert!(n != m);
        m.push_back(1);
        assert_eq!(&n, &m);
    }

    #[test]
    fn test_fuzz() {
        for 25.times {
            fuzz_test(3);
            fuzz_test(16);
            fuzz_test(189);
        }
    }

    #[cfg(test)]
    fn fuzz_test(sz: int) {
        let mut m = DList::new::<int>();
        let mut v = ~[];
        for int::range(0i, sz) |i| {
            check_links(&m);
            let r: u8 = rand::random();
            match r % 6 {
                0 => {
                    m.pop_back();
                    if v.len() > 0 { v.pop(); }
                }
                1 => {
                    m.pop_front();
                    if v.len() > 0 { v.shift(); }
                }
                2 | 4 =>  {
                    m.push_front(-i);
                    v.unshift(-i);
                }
                3 | 5 | _ => {
                    m.push_back(i);
                    v.push(i);
                }
            }
        }

        check_links(&m);

        let mut i = 0u;
        for m.consume_iter().zip(v.iter()).advance |(a, &b)| {
            i += 1;
            assert_eq!(a, b);
        }
        assert_eq!(i, v.len());
    }

    #[bench]
    fn bench_collect_into(b: &mut test::BenchHarness) {
        let v = &[0, ..64];
        do b.iter {
            let _: DList<int> = v.iter().transform(|x| *x).collect();
        }
    }
    #[bench]
    fn bench_collect_into_vec(b: &mut test::BenchHarness) {
        let v = &[0, ..64];
        do b.iter {
            let _: ~[int] = v.iter().transform(|&x|x).collect();
        }
    }

    #[bench]
    fn bench_push_front(b: &mut test::BenchHarness) {
        let mut m = DList::new::<int>();
        do b.iter {
            m.push_front(0);
        }
    }
    #[bench]
    fn bench_push_front_vec_size10(b: &mut test::BenchHarness) {
        let mut m = ~[0, ..10];
        do b.iter {
            m.unshift(0);
            m.pop(); // to keep it fair, dont' grow the vec
        }
    }

    #[bench]
    fn bench_push_back(b: &mut test::BenchHarness) {
        let mut m = DList::new::<int>();
        do b.iter {
            m.push_back(0);
        }
    }
    #[bench]
    fn bench_push_back_vec(b: &mut test::BenchHarness) {
        let mut m = ~[];
        do b.iter {
            m.push(0);
        }
    }

    #[bench]
    fn bench_push_back_pop_back(b: &mut test::BenchHarness) {
        let mut m = DList::new::<int>();
        do b.iter {
            m.push_back(0);
            m.pop_back();
        }
    }
    #[bench]
    fn bench_push_back_pop_back_vec(b: &mut test::BenchHarness) {
        let mut m = ~[];
        do b.iter {
            m.push(0);
            m.pop();
        }
    }

    #[bench]
    fn bench_iter(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let m: DList<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            assert!(m.iter().len_() == 128);
        }
    }
    #[bench]
    fn bench_iter_mut(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let mut m: DList<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            assert!(m.mut_iter().len_() == 128);
        }
    }
    #[bench]
    fn bench_iter_rev(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let m: DList<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            assert!(m.rev_iter().len_() == 128);
        }
    }
    #[bench]
    fn bench_iter_mut_rev(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        let mut m: DList<int> = v.iter().transform(|&x|x).collect();
        do b.iter {
            assert!(m.mut_rev_iter().len_() == 128);
        }
    }
    #[bench]
    fn bench_iter_vec(b: &mut test::BenchHarness) {
        let v = &[0, ..128];
        do b.iter {
            for v.iter().advance |_| {}
        }
    }
}

