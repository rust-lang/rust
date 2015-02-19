// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
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
//! The `LinkedList` allows pushing and popping elements at either end and is thus
//! efficiently usable as a double-ended queue.

// LinkedList is constructed like a singly-linked list over the field `next`.
// including the last link being None; each Node owns its `next` field.
//
// Backlinks over LinkedList::prev are raw pointers that form a full chain in
// the reverse direction.

#![stable(feature = "rust1", since = "1.0.0")]

use core::prelude::*;

use alloc::boxed::Box;
use core::cmp::Ordering;
use core::default::Default;
use core::fmt;
use core::hash::{Hasher, Hash};
#[cfg(stage0)]
use core::hash::Writer;
use core::iter::{self, FromIterator, IntoIterator};
use core::mem;
use core::ptr;

#[deprecated(since = "1.0.0", reason = "renamed to LinkedList")]
#[unstable(feature = "collections")]
pub use LinkedList as DList;

/// A doubly-linked list.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LinkedList<T> {
    length: usize,
    list_head: Link<T>,
    list_tail: Rawlink<Node<T>>,
}

type Link<T> = Option<Box<Node<T>>>;

struct Rawlink<T> {
    p: *mut T,
}

impl<T> Copy for Rawlink<T> {}
unsafe impl<T:'static+Send> Send for Rawlink<T> {}
unsafe impl<T:Send+Sync> Sync for Rawlink<T> {}

struct Node<T> {
    next: Link<T>,
    prev: Rawlink<Node<T>>,
    value: T,
}

/// An iterator over references to the items of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T:'a> {
    head: &'a Link<T>,
    tail: Rawlink<Node<T>>,
    nelem: usize,
}

// FIXME #19839: deriving is too aggressive on the bounds (T doesn't need to be Clone).
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Iter<'a, T> {
        Iter {
            head: self.head.clone(),
            tail: self.tail,
            nelem: self.nelem,
        }
    }
}

/// An iterator over mutable references to the items of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T:'a> {
    list: &'a mut LinkedList<T>,
    head: Rawlink<Node<T>>,
    tail: Rawlink<Node<T>>,
    nelem: usize,
}

/// An iterator over mutable references to the items of a `LinkedList`.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    list: LinkedList<T>
}

/// Rawlink is a type like Option<T> but for holding a raw pointer
impl<T> Rawlink<T> {
    /// Like Option::None for Rawlink
    fn none() -> Rawlink<T> {
        Rawlink{p: ptr::null_mut()}
    }

    /// Like Option::Some for Rawlink
    fn some(n: &mut T) -> Rawlink<T> {
        Rawlink{p: n}
    }

    /// Convert the `Rawlink` into an Option value
    fn resolve_immut<'a>(&self) -> Option<&'a T> {
        unsafe {
            mem::transmute(self.p.as_ref())
        }
    }

    /// Convert the `Rawlink` into an Option value
    fn resolve<'a>(&mut self) -> Option<&'a mut T> {
        if self.p.is_null() {
            None
        } else {
            Some(unsafe { mem::transmute(self.p) })
        }
    }

    /// Return the `Rawlink` and replace with `Rawlink::none()`
    fn take(&mut self) -> Rawlink<T> {
        mem::replace(self, Rawlink::none())
    }
}

impl<T> Clone for Rawlink<T> {
    #[inline]
    fn clone(&self) -> Rawlink<T> {
        Rawlink{p: self.p}
    }
}

impl<T> Node<T> {
    fn new(v: T) -> Node<T> {
        Node{value: v, next: None, prev: Rawlink::none()}
    }
}

/// Set the .prev field on `next`, then return `Some(next)`
fn link_with_prev<T>(mut next: Box<Node<T>>, prev: Rawlink<Node<T>>)
                  -> Link<T> {
    next.prev = prev;
    Some(next)
}

// private methods
impl<T> LinkedList<T> {
    /// Add a Node first in the list
    #[inline]
    fn push_front_node(&mut self, mut new_head: Box<Node<T>>) {
        match self.list_head {
            None => {
                self.list_tail = Rawlink::some(&mut *new_head);
                self.list_head = link_with_prev(new_head, Rawlink::none());
            }
            Some(ref mut head) => {
                new_head.prev = Rawlink::none();
                head.prev = Rawlink::some(&mut *new_head);
                mem::swap(head, &mut new_head);
                head.next = Some(new_head);
            }
        }
        self.length += 1;
    }

    /// Remove the first Node and return it, or None if the list is empty
    #[inline]
    fn pop_front_node(&mut self) -> Option<Box<Node<T>>> {
        self.list_head.take().map(|mut front_node| {
            self.length -= 1;
            match front_node.next.take() {
                Some(node) => self.list_head = link_with_prev(node, Rawlink::none()),
                None => self.list_tail = Rawlink::none()
            }
            front_node
        })
    }

    /// Add a Node last in the list
    #[inline]
    fn push_back_node(&mut self, mut new_tail: Box<Node<T>>) {
        match self.list_tail.resolve() {
            None => return self.push_front_node(new_tail),
            Some(tail) => {
                self.list_tail = Rawlink::some(&mut *new_tail);
                tail.next = link_with_prev(new_tail, Rawlink::some(tail));
            }
        }
        self.length += 1;
    }

    /// Remove the last Node and return it, or None if the list is empty
    #[inline]
    fn pop_back_node(&mut self) -> Option<Box<Node<T>>> {
        self.list_tail.resolve().map_or(None, |tail| {
            self.length -= 1;
            self.list_tail = tail.prev;
            match tail.prev.resolve() {
                None => self.list_head.take(),
                Some(tail_prev) => tail_prev.next.take()
            }
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for LinkedList<T> {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> LinkedList<T> { LinkedList::new() }
}

impl<T> LinkedList<T> {
    /// Creates an empty `LinkedList`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> LinkedList<T> {
        LinkedList{list_head: None, list_tail: Rawlink::none(), length: 0}
    }

    /// Moves all elements from `other` to the end of the list.
    ///
    /// This reuses all the nodes from `other` and moves them into `self`. After
    /// this operation, `other` becomes empty.
    ///
    /// This operation should compute in O(1) time and O(1) memory.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut a = LinkedList::new();
    /// let mut b = LinkedList::new();
    /// a.push_back(1);
    /// a.push_back(2);
    /// b.push_back(3);
    /// b.push_back(4);
    ///
    /// a.append(&mut b);
    ///
    /// for e in a.iter() {
    ///     println!("{}", e); // prints 1, then 2, then 3, then 4
    /// }
    /// println!("{}", b.len()); // prints 0
    /// ```
    pub fn append(&mut self, other: &mut LinkedList<T>) {
        match self.list_tail.resolve() {
            None => {
                self.length = other.length;
                self.list_head = other.list_head.take();
                self.list_tail = other.list_tail.take();
            },
            Some(tail) => {
                // Carefully empty `other`.
                let o_tail = other.list_tail.take();
                let o_length = other.length;
                match other.list_head.take() {
                    None => return,
                    Some(node) => {
                        tail.next = link_with_prev(node, self.list_tail);
                        self.list_tail = o_tail;
                        self.length += o_length;
                    }
                }
            }
        }
        other.length = 0;
    }

    /// Provides a forward iterator.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter{nelem: self.len(), head: &self.list_head, tail: self.list_tail}
    }

    /// Provides a forward iterator with mutable references.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        let head_raw = match self.list_head {
            Some(ref mut h) => Rawlink::some(&mut **h),
            None => Rawlink::none(),
        };
        IterMut{
            nelem: self.len(),
            head: head_raw,
            tail: self.list_tail,
            list: self
        }
    }

    /// Consumes the list into an iterator yielding elements by value.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn into_iter(self) -> IntoIter<T> {
        IntoIter{list: self}
    }

    /// Returns `true` if the `LinkedList` is empty.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    /// assert!(dl.is_empty());
    ///
    /// dl.push_front("foo");
    /// assert!(!dl.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn is_empty(&self) -> bool {
        self.list_head.is_none()
    }

    /// Returns the length of the `LinkedList`.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    ///
    /// dl.push_front(2);
    /// assert_eq!(dl.len(), 1);
    ///
    /// dl.push_front(1);
    /// assert_eq!(dl.len(), 2);
    ///
    /// dl.push_back(3);
    /// assert_eq!(dl.len(), 3);
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.length
    }

    /// Removes all elements from the `LinkedList`.
    ///
    /// This operation should compute in O(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    ///
    /// dl.push_front(2);
    /// dl.push_front(1);
    /// assert_eq!(dl.len(), 2);
    /// assert_eq!(dl.front(), Some(&1));
    ///
    /// dl.clear();
    /// assert_eq!(dl.len(), 0);
    /// assert_eq!(dl.front(), None);
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        *self = LinkedList::new()
    }

    /// Provides a reference to the front element, or `None` if the list is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    /// assert_eq!(dl.front(), None);
    ///
    /// dl.push_front(1);
    /// assert_eq!(dl.front(), Some(&1));
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front(&self) -> Option<&T> {
        self.list_head.as_ref().map(|head| &head.value)
    }

    /// Provides a mutable reference to the front element, or `None` if the list
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    /// assert_eq!(dl.front(), None);
    ///
    /// dl.push_front(1);
    /// assert_eq!(dl.front(), Some(&1));
    ///
    /// match dl.front_mut() {
    ///     None => {},
    ///     Some(x) => *x = 5,
    /// }
    /// assert_eq!(dl.front(), Some(&5));
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.list_head.as_mut().map(|head| &mut head.value)
    }

    /// Provides a reference to the back element, or `None` if the list is
    /// empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    /// assert_eq!(dl.back(), None);
    ///
    /// dl.push_back(1);
    /// assert_eq!(dl.back(), Some(&1));
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back(&self) -> Option<&T> {
        self.list_tail.resolve_immut().as_ref().map(|tail| &tail.value)
    }

    /// Provides a mutable reference to the back element, or `None` if the list
    /// is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    /// assert_eq!(dl.back(), None);
    ///
    /// dl.push_back(1);
    /// assert_eq!(dl.back(), Some(&1));
    ///
    /// match dl.back_mut() {
    ///     None => {},
    ///     Some(x) => *x = 5,
    /// }
    /// assert_eq!(dl.back(), Some(&5));
    ///
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.list_tail.resolve().map(|tail| &mut tail.value)
    }

    /// Adds an element first in the list.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut dl = LinkedList::new();
    ///
    /// dl.push_front(2);
    /// assert_eq!(dl.front().unwrap(), &2);
    ///
    /// dl.push_front(1);
    /// assert_eq!(dl.front().unwrap(), &1);
    ///
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_front(&mut self, elt: T) {
        self.push_front_node(box Node::new(elt))
    }

    /// Removes the first element and returns it, or `None` if the list is
    /// empty.
    ///
    /// This operation should compute in O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut d = LinkedList::new();
    /// assert_eq!(d.pop_front(), None);
    ///
    /// d.push_front(1);
    /// d.push_front(3);
    /// assert_eq!(d.pop_front(), Some(3));
    /// assert_eq!(d.pop_front(), Some(1));
    /// assert_eq!(d.pop_front(), None);
    ///
    /// ```
    ///
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop_front(&mut self) -> Option<T> {
        self.pop_front_node().map(|box Node{value, ..}| value)
    }

    /// Appends an element to the back of a list
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut d = LinkedList::new();
    /// d.push_back(1);
    /// d.push_back(3);
    /// assert_eq!(3, *d.back().unwrap());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_back(&mut self, elt: T) {
        self.push_back_node(box Node::new(elt))
    }

    /// Removes the last element from a list and returns it, or `None` if
    /// it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut d = LinkedList::new();
    /// assert_eq!(d.pop_back(), None);
    /// d.push_back(1);
    /// d.push_back(3);
    /// assert_eq!(d.pop_back(), Some(3));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop_back(&mut self) -> Option<T> {
        self.pop_back_node().map(|box Node{value, ..}| value)
    }

    /// Splits the list into two at the given index. Returns everything after the given index,
    /// including the index.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
    ///
    /// This operation should compute in O(n) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut d = LinkedList::new();
    ///
    /// d.push_front(1);
    /// d.push_front(2);
    /// d.push_front(3);
    ///
    /// let mut splitted = d.split_off(2);
    ///
    /// assert_eq!(splitted.pop_front(), Some(1));
    /// assert_eq!(splitted.pop_front(), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn split_off(&mut self, at: usize) -> LinkedList<T> {
        let len = self.len();
        assert!(at <= len, "Cannot split off at a nonexistent index");
        if at == 0 {
            return mem::replace(self, LinkedList::new());
        } else if at == len {
            return LinkedList::new();
        }

        // Below, we iterate towards the `i-1`th node, either from the start or the end,
        // depending on which would be faster.
        let mut split_node = if at - 1 <= len - 1 - (at - 1) {
            let mut iter = self.iter_mut();
            // instead of skipping using .skip() (which creates a new struct),
            // we skip manually so we can access the head field without
            // depending on implementation details of Skip
            for _ in 0..at - 1 {
                iter.next();
            }
            iter.head
        }  else {
            // better off starting from the end
            let mut iter = self.iter_mut();
            for _ in 0..len - 1 - (at - 1) {
                iter.next_back();
            }
            iter.tail
        };

        let mut splitted_list = LinkedList {
            list_head: None,
            list_tail: self.list_tail,
            length: len - at
        };

        mem::swap(&mut split_node.resolve().unwrap().next, &mut splitted_list.list_head);
        self.list_tail = split_node;
        self.length = at;

        splitted_list
    }
}

#[unsafe_destructor]
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        // Dissolve the linked_list in backwards direction
        // Just dropping the list_head can lead to stack exhaustion
        // when length is >> 1_000_000
        let mut tail = self.list_tail;
        loop {
            match tail.resolve() {
                None => break,
                Some(prev) => {
                    prev.next.take(); // release Box<Node<T>>
                    tail = prev.prev;
                }
            }
        }
        self.length = 0;
        self.list_head = None;
        self.list_tail = Rawlink::none();
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<&'a A> {
        if self.nelem == 0 {
            return None;
        }
        self.head.as_ref().map(|head| {
            self.nelem -= 1;
            self.head = &head.next;
            &head.value
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nelem, Some(self.nelem))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for Iter<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a A> {
        if self.nelem == 0 {
            return None;
        }
        self.tail.resolve_immut().as_ref().map(|prev| {
            self.nelem -= 1;
            self.tail = prev.prev;
            &prev.value
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> ExactSizeIterator for Iter<'a, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<&'a mut A> {
        if self.nelem == 0 {
            return None;
        }
        self.head.resolve().map(|next| {
            self.nelem -= 1;
            self.head = match next.next {
                Some(ref mut node) => Rawlink::some(&mut **node),
                None => Rawlink::none(),
            };
            &mut next.value
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.nelem, Some(self.nelem))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> DoubleEndedIterator for IterMut<'a, A> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut A> {
        if self.nelem == 0 {
            return None;
        }
        self.tail.resolve().map(|prev| {
            self.nelem -= 1;
            self.tail = prev.prev;
            &mut prev.value
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> ExactSizeIterator for IterMut<'a, A> {}

// private methods for IterMut
impl<'a, A> IterMut<'a, A> {
    fn insert_next_node(&mut self, mut ins_node: Box<Node<A>>) {
        // Insert before `self.head` so that it is between the
        // previously yielded element and self.head.
        //
        // The inserted node will not appear in further iteration.
        match self.head.resolve() {
            None => { self.list.push_back_node(ins_node); }
            Some(node) => {
                let prev_node = match node.prev.resolve() {
                    None => return self.list.push_front_node(ins_node),
                    Some(prev) => prev,
                };
                let node_own = prev_node.next.take().unwrap();
                ins_node.next = link_with_prev(node_own, Rawlink::some(&mut *ins_node));
                prev_node.next = link_with_prev(ins_node, Rawlink::some(prev_node));
                self.list.length += 1;
            }
        }
    }
}

impl<'a, A> IterMut<'a, A> {
    /// Inserts `elt` just after the element most recently returned by `.next()`.
    /// The inserted element does not appear in the iteration.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut list: LinkedList<_> = vec![1, 3, 4].into_iter().collect();
    ///
    /// {
    ///     let mut it = list.iter_mut();
    ///     assert_eq!(it.next().unwrap(), &1);
    ///     // insert `2` after `1`
    ///     it.insert_next(2);
    /// }
    /// {
    ///     let vec: Vec<_> = list.into_iter().collect();
    ///     assert_eq!(vec, vec![1, 2, 3, 4]);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "this is probably better handled by a cursor type -- we'll see")]
    pub fn insert_next(&mut self, elt: A) {
        self.insert_next_node(box Node::new(elt))
    }

    /// Provides a reference to the next element, without changing the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut list: LinkedList<_> = vec![1, 2, 3].into_iter().collect();
    ///
    /// let mut it = list.iter_mut();
    /// assert_eq!(it.next().unwrap(), &1);
    /// assert_eq!(it.peek_next().unwrap(), &2);
    /// // We just peeked at 2, so it was not consumed from the iterator.
    /// assert_eq!(it.next().unwrap(), &2);
    /// ```
    #[inline]
    #[unstable(feature = "collections",
               reason = "this is probably better handled by a cursor type -- we'll see")]
    pub fn peek_next(&mut self) -> Option<&mut A> {
        if self.nelem == 0 {
            return None
        }
        self.head.resolve().map(|head| &mut head.value)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Iterator for IntoIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> { self.list.pop_front() }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.length, Some(self.list.length))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> { self.list.pop_back() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> FromIterator<A> for LinkedList<A> {
    fn from_iter<T: IntoIterator<Item=A>>(iter: T) -> LinkedList<A> {
        let mut ret = DList::new();
        ret.extend(iter);
        ret
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        self.into_iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(mut self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Extend<A> for LinkedList<A> {
    fn extend<T: IntoIterator<Item=A>>(&mut self, iter: T) {
        for elt in iter { self.push_back(elt); }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialEq> PartialEq for LinkedList<A> {
    fn eq(&self, other: &LinkedList<A>) -> bool {
        self.len() == other.len() &&
            iter::order::eq(self.iter(), other.iter())
    }

    fn ne(&self, other: &LinkedList<A>) -> bool {
        self.len() != other.len() ||
            iter::order::ne(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Eq> Eq for LinkedList<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialOrd> PartialOrd for LinkedList<A> {
    fn partial_cmp(&self, other: &LinkedList<A>) -> Option<Ordering> {
        iter::order::partial_cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Ord> Ord for LinkedList<A> {
    #[inline]
    fn cmp(&self, other: &LinkedList<A>) -> Ordering {
        iter::order::cmp(self.iter(), other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> Clone for LinkedList<A> {
    fn clone(&self) -> LinkedList<A> {
        self.iter().cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: fmt::Debug> fmt::Debug for LinkedList<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "LinkedList ["));

        for (i, e) in self.iter().enumerate() {
            if i != 0 { try!(write!(f, ", ")); }
            try!(write!(f, "{:?}", *e));
        }

        write!(f, "]")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(stage0)]
impl<S: Writer + Hasher, A: Hash<S>> Hash<S> for LinkedList<A> {
    fn hash(&self, state: &mut S) {
        self.len().hash(state);
        for elt in self {
            elt.hash(state);
        }
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg(not(stage0))]
impl<A: Hash> Hash for LinkedList<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for elt in self {
            elt.hash(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use prelude::*;
    use std::rand;
    use std::hash::{self, SipHasher};
    use std::thread;
    use test::Bencher;
    use test;

    use super::{LinkedList, Node};

    pub fn check_links<T>(list: &LinkedList<T>) {
        let mut len = 0;
        let mut last_ptr: Option<&Node<T>> = None;
        let mut node_ptr: &Node<T>;
        match list.list_head {
            None => { assert_eq!(0, list.length); return }
            Some(ref node) => node_ptr = &**node,
        }
        loop {
            match (last_ptr, node_ptr.prev.resolve_immut()) {
                (None   , None      ) => {}
                (None   , _         ) => panic!("prev link for list_head"),
                (Some(p), Some(pptr)) => {
                    assert_eq!(p as *const Node<T>, pptr as *const Node<T>);
                }
                _ => panic!("prev link is none, not good"),
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

    #[test]
    fn test_basic() {
        let mut m = LinkedList::new();
        assert_eq!(m.pop_front(), None);
        assert_eq!(m.pop_back(), None);
        assert_eq!(m.pop_front(), None);
        m.push_front(box 1);
        assert_eq!(m.pop_front(), Some(box 1));
        m.push_back(box 2);
        m.push_back(box 3);
        assert_eq!(m.len(), 2);
        assert_eq!(m.pop_front(), Some(box 2));
        assert_eq!(m.pop_front(), Some(box 3));
        assert_eq!(m.len(), 0);
        assert_eq!(m.pop_front(), None);
        m.push_back(box 1);
        m.push_back(box 3);
        m.push_back(box 5);
        m.push_back(box 7);
        assert_eq!(m.pop_front(), Some(box 1));

        let mut n = LinkedList::new();
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
    fn generate_test() -> LinkedList<i32> {
        list_from(&[0,1,2,3,4,5,6])
    }

    #[cfg(test)]
    fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
        v.iter().cloned().collect()
    }

    #[test]
    fn test_append() {
        // Empty to empty
        {
            let mut m = LinkedList::<i32>::new();
            let mut n = LinkedList::new();
            m.append(&mut n);
            check_links(&m);
            assert_eq!(m.len(), 0);
            assert_eq!(n.len(), 0);
        }
        // Non-empty to empty
        {
            let mut m = LinkedList::new();
            let mut n = LinkedList::new();
            n.push_back(2);
            m.append(&mut n);
            check_links(&m);
            assert_eq!(m.len(), 1);
            assert_eq!(m.pop_back(), Some(2));
            assert_eq!(n.len(), 0);
            check_links(&m);
        }
        // Empty to non-empty
        {
            let mut m = LinkedList::new();
            let mut n = LinkedList::new();
            m.push_back(2);
            m.append(&mut n);
            check_links(&m);
            assert_eq!(m.len(), 1);
            assert_eq!(m.pop_back(), Some(2));
            check_links(&m);
        }

        // Non-empty to non-empty
        let v = vec![1,2,3,4,5];
        let u = vec![9,8,1,2,3,4,5];
        let mut m = list_from(&v);
        let mut n = list_from(&u);
        m.append(&mut n);
        check_links(&m);
        let mut sum = v;
        sum.push_all(&u);
        assert_eq!(sum.len(), m.len());
        for elt in sum {
            assert_eq!(m.pop_front(), Some(elt))
        }
        assert_eq!(n.len(), 0);
        // let's make sure it's working properly, since we
        // did some direct changes to private members
        n.push_back(3);
        assert_eq!(n.len(), 1);
        assert_eq!(n.pop_front(), Some(3));
        check_links(&n);
    }

    #[test]
    fn test_split_off() {
        // singleton
        {
            let mut m = LinkedList::new();
            m.push_back(1);

            let p = m.split_off(0);
            assert_eq!(m.len(), 0);
            assert_eq!(p.len(), 1);
            assert_eq!(p.back(), Some(&1));
            assert_eq!(p.front(), Some(&1));
        }

        // not singleton, forwards
        {
            let u = vec![1,2,3,4,5];
            let mut m = list_from(&u);
            let mut n = m.split_off(2);
            assert_eq!(m.len(), 2);
            assert_eq!(n.len(), 3);
            for elt in 1..3 {
                assert_eq!(m.pop_front(), Some(elt));
            }
            for elt in 3..6 {
                assert_eq!(n.pop_front(), Some(elt));
            }
        }
        // not singleton, backwards
        {
            let u = vec![1,2,3,4,5];
            let mut m = list_from(&u);
            let mut n = m.split_off(4);
            assert_eq!(m.len(), 4);
            assert_eq!(n.len(), 1);
            for elt in 1..5 {
                assert_eq!(m.pop_front(), Some(elt));
            }
            for elt in 5..6 {
                assert_eq!(n.pop_front(), Some(elt));
            }
        }

        // no-op on the last index
        {
            let mut m = LinkedList::new();
            m.push_back(1);

            let p = m.split_off(1);
            assert_eq!(m.len(), 1);
            assert_eq!(p.len(), 0);
            assert_eq!(m.back(), Some(&1));
            assert_eq!(m.front(), Some(&1));
        }

    }

    #[test]
    fn test_iterator() {
        let m = generate_test();
        for (i, elt) in m.iter().enumerate() {
            assert_eq!(i as i32, *elt);
        }
        let mut n = LinkedList::new();
        assert_eq!(n.iter().next(), None);
        n.push_front(4);
        let mut it = n.iter();
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &4);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_iterator_clone() {
        let mut n = LinkedList::new();
        n.push_back(2);
        n.push_back(3);
        n.push_back(4);
        let mut it = n.iter();
        it.next();
        let mut jt = it.clone();
        assert_eq!(it.next(), jt.next());
        assert_eq!(it.next_back(), jt.next_back());
        assert_eq!(it.next(), jt.next());
    }

    #[test]
    fn test_iterator_double_end() {
        let mut n = LinkedList::new();
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
        for (i, elt) in m.iter().rev().enumerate() {
            assert_eq!((6 - i) as i32, *elt);
        }
        let mut n = LinkedList::new();
        assert_eq!(n.iter().rev().next(), None);
        n.push_front(4);
        let mut it = n.iter().rev();
        assert_eq!(it.size_hint(), (1, Some(1)));
        assert_eq!(it.next().unwrap(), &4);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.next(), None);
    }

    #[test]
    fn test_mut_iter() {
        let mut m = generate_test();
        let mut len = m.len();
        for (i, elt) in m.iter_mut().enumerate() {
            assert_eq!(i as i32, *elt);
            len -= 1;
        }
        assert_eq!(len, 0);
        let mut n = LinkedList::new();
        assert!(n.iter_mut().next().is_none());
        n.push_front(4);
        n.push_back(5);
        let mut it = n.iter_mut();
        assert_eq!(it.size_hint(), (2, Some(2)));
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }

    #[test]
    fn test_iterator_mut_double_end() {
        let mut n = LinkedList::new();
        assert!(n.iter_mut().next_back().is_none());
        n.push_front(4);
        n.push_front(5);
        n.push_front(6);
        let mut it = n.iter_mut();
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
            let mut it = m.iter_mut();
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
        assert_eq!(m.into_iter().collect::<Vec<_>>(), vec![-2,0,1,2,3,4,5,6,7,8,9,0,1]);
    }

    #[test]
    fn test_mut_rev_iter() {
        let mut m = generate_test();
        for (i, elt) in m.iter_mut().rev().enumerate() {
            assert_eq!((6 - i) as i32, *elt);
        }
        let mut n = LinkedList::new();
        assert!(n.iter_mut().rev().next().is_none());
        n.push_front(4);
        let mut it = n.iter_mut().rev();
        assert!(it.next().is_some());
        assert!(it.next().is_none());
    }

    #[test]
    fn test_send() {
        let n = list_from(&[1,2,3]);
        thread::spawn(move || {
            check_links(&n);
            let a: &[_] = &[&1,&2,&3];
            assert_eq!(a, n.iter().collect::<Vec<_>>());
        }).join().ok().unwrap();
    }

    #[test]
    fn test_eq() {
        let mut n = list_from(&[]);
        let mut m = list_from(&[]);
        assert!(n == m);
        n.push_front(1);
        assert!(n != m);
        m.push_back(1);
        assert!(n == m);

        let n = list_from(&[2,3,4]);
        let m = list_from(&[1,2,3]);
        assert!(n != m);
    }

    #[test]
    fn test_hash() {
      let mut x = LinkedList::new();
      let mut y = LinkedList::new();

      assert!(hash::hash::<_, SipHasher>(&x) == hash::hash::<_, SipHasher>(&y));

      x.push_back(1);
      x.push_back(2);
      x.push_back(3);

      y.push_front(3);
      y.push_front(2);
      y.push_front(1);

      assert!(hash::hash::<_, SipHasher>(&x) == hash::hash::<_, SipHasher>(&y));
    }

    #[test]
    fn test_ord() {
        let n = list_from(&[]);
        let m = list_from(&[1,2,3]);
        assert!(n < m);
        assert!(m > n);
        assert!(n <= n);
        assert!(n >= n);
    }

    #[test]
    fn test_ord_nan() {
        let nan = 0.0f64/0.0;
        let n = list_from(&[nan]);
        let m = list_from(&[nan]);
        assert!(!(n < m));
        assert!(!(n > m));
        assert!(!(n <= m));
        assert!(!(n >= m));

        let n = list_from(&[nan]);
        let one = list_from(&[1.0f64]);
        assert!(!(n < one));
        assert!(!(n > one));
        assert!(!(n <= one));
        assert!(!(n >= one));

        let u = list_from(&[1.0f64,2.0,nan]);
        let v = list_from(&[1.0f64,2.0,3.0]);
        assert!(!(u < v));
        assert!(!(u > v));
        assert!(!(u <= v));
        assert!(!(u >= v));

        let s = list_from(&[1.0f64,2.0,4.0,2.0]);
        let t = list_from(&[1.0f64,2.0,3.0,2.0]);
        assert!(!(s < t));
        assert!(s > one);
        assert!(!(s <= one));
        assert!(s >= one);
    }

    #[test]
    fn test_fuzz() {
        for _ in 0..25 {
            fuzz_test(3);
            fuzz_test(16);
            fuzz_test(189);
        }
    }

    #[test]
    fn test_show() {
        let list: LinkedList<_> = (0..10).collect();
        assert_eq!(format!("{:?}", list), "LinkedList [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");

        let list: LinkedList<_> = vec!["just", "one", "test", "more"].iter().cloned().collect();
        assert_eq!(format!("{:?}", list), "LinkedList [\"just\", \"one\", \"test\", \"more\"]");
    }

    #[cfg(test)]
    fn fuzz_test(sz: i32) {
        let mut m: LinkedList<_> = LinkedList::new();
        let mut v = vec![];
        for i in 0..sz {
            check_links(&m);
            let r: u8 = rand::random();
            match r % 6 {
                0 => {
                    m.pop_back();
                    v.pop();
                }
                1 => {
                    if !v.is_empty() {
                        m.pop_front();
                        v.remove(0);
                    }
                }
                2 | 4 =>  {
                    m.push_front(-i);
                    v.insert(0, -i);
                }
                3 | 5 | _ => {
                    m.push_back(i);
                    v.push(i);
                }
            }
        }

        check_links(&m);

        let mut i = 0;
        for (a, &b) in m.into_iter().zip(v.iter()) {
            i += 1;
            assert_eq!(a, b);
        }
        assert_eq!(i, v.len());
    }

    #[bench]
    fn bench_collect_into(b: &mut test::Bencher) {
        let v = &[0; 64];
        b.iter(|| {
            let _: LinkedList<_> = v.iter().cloned().collect();
        })
    }

    #[bench]
    fn bench_push_front(b: &mut test::Bencher) {
        let mut m: LinkedList<_> = LinkedList::new();
        b.iter(|| {
            m.push_front(0);
        })
    }

    #[bench]
    fn bench_push_back(b: &mut test::Bencher) {
        let mut m: LinkedList<_> = LinkedList::new();
        b.iter(|| {
            m.push_back(0);
        })
    }

    #[bench]
    fn bench_push_back_pop_back(b: &mut test::Bencher) {
        let mut m: LinkedList<_> = LinkedList::new();
        b.iter(|| {
            m.push_back(0);
            m.pop_back();
        })
    }

    #[bench]
    fn bench_push_front_pop_front(b: &mut test::Bencher) {
        let mut m: LinkedList<_> = LinkedList::new();
        b.iter(|| {
            m.push_front(0);
            m.pop_front();
        })
    }

    #[bench]
    fn bench_iter(b: &mut test::Bencher) {
        let v = &[0; 128];
        let m: LinkedList<_> = v.iter().cloned().collect();
        b.iter(|| {
            assert!(m.iter().count() == 128);
        })
    }
    #[bench]
    fn bench_iter_mut(b: &mut test::Bencher) {
        let v = &[0; 128];
        let mut m: LinkedList<_> = v.iter().cloned().collect();
        b.iter(|| {
            assert!(m.iter_mut().count() == 128);
        })
    }
    #[bench]
    fn bench_iter_rev(b: &mut test::Bencher) {
        let v = &[0; 128];
        let m: LinkedList<_> = v.iter().cloned().collect();
        b.iter(|| {
            assert!(m.iter().rev().count() == 128);
        })
    }
    #[bench]
    fn bench_iter_mut_rev(b: &mut test::Bencher) {
        let v = &[0; 128];
        let mut m: LinkedList<_> = v.iter().cloned().collect();
        b.iter(|| {
            assert!(m.iter_mut().rev().count() == 128);
        })
    }
}
