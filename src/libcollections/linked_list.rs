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

use alloc::boxed::Box;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hasher, Hash};
use core::iter::FromIterator;
use core::mem;
use core::ptr::Shared;

/// A doubly-linked list.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LinkedList<T> {
    length: usize,
    list_head: Link<T>,
    list_tail: Rawlink<Node<T>>,
}

type Link<T> = Option<Box<Node<T>>>;

struct Rawlink<T> {
    p: Option<Shared<T>>,
}

impl<T> Copy for Rawlink<T> {}
unsafe impl<T: Send> Send for Rawlink<T> {}
unsafe impl<T: Sync> Sync for Rawlink<T> {}

struct Node<T> {
    next: Link<T>,
    prev: Rawlink<Node<T>>,
    value: T,
}

/// An iterator over references to the items of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    head: &'a Link<T>,
    tail: Rawlink<Node<T>>,
    nelem: usize,
}

// FIXME #19839: deriving is too aggressive on the bounds (T doesn't need to be Clone).
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Iter {
            head: self.head.clone(),
            tail: self.tail,
            nelem: self.nelem,
        }
    }
}

/// An iterator over mutable references to the items of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    list: &'a mut LinkedList<T>,
    head: Rawlink<Node<T>>,
    tail: Rawlink<Node<T>>,
    nelem: usize,
}

/// An iterator over mutable references to the items of a `LinkedList`.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    list: LinkedList<T>,
}

/// Rawlink is a type like Option<T> but for holding a raw pointer
impl<T> Rawlink<T> {
    /// Like Option::None for Rawlink
    fn none() -> Self {
        Rawlink { p: None }
    }

    /// Like Option::Some for Rawlink
    fn some(n: &mut T) -> Self {
        unsafe { Rawlink { p: Some(Shared::new(n)) } }
    }

    /// Convert the `Rawlink` into an Option value
    ///
    /// **unsafe** because:
    ///
    /// - Dereference of raw pointer.
    /// - Returns reference of arbitrary lifetime.
    unsafe fn resolve<'a>(&self) -> Option<&'a T> {
        self.p.map(|p| &**p)
    }

    /// Convert the `Rawlink` into an Option value
    ///
    /// **unsafe** because:
    ///
    /// - Dereference of raw pointer.
    /// - Returns reference of arbitrary lifetime.
    unsafe fn resolve_mut<'a>(&mut self) -> Option<&'a mut T> {
        self.p.map(|p| &mut **p)
    }

    /// Return the `Rawlink` and replace with `Rawlink::none()`
    fn take(&mut self) -> Self {
        mem::replace(self, Rawlink::none())
    }
}

impl<'a, T> From<&'a mut Link<T>> for Rawlink<Node<T>> {
    fn from(node: &'a mut Link<T>) -> Self {
        match node.as_mut() {
            None => Rawlink::none(),
            Some(ptr) => Rawlink::some(ptr),
        }
    }
}

impl<T> Clone for Rawlink<T> {
    #[inline]
    fn clone(&self) -> Self {
        Rawlink { p: self.p }
    }
}

impl<T> Node<T> {
    fn new(v: T) -> Self {
        Node {
            value: v,
            next: None,
            prev: Rawlink::none(),
        }
    }

    /// Update the `prev` link on `next`, then set self's next pointer.
    ///
    /// `self.next` should be `None` when you call this
    /// (otherwise a Node is probably being dropped by mistake).
    fn set_next(&mut self, mut next: Box<Node<T>>) {
        debug_assert!(self.next.is_none());
        next.prev = Rawlink::some(self);
        self.next = Some(next);
    }
}

/// Clear the .prev field on `next`, then return `Some(next)`
fn link_no_prev<T>(mut next: Box<Node<T>>) -> Link<T> {
    next.prev = Rawlink::none();
    Some(next)
}

// private methods
impl<T> LinkedList<T> {
    /// Add a Node first in the list
    #[inline]
    fn push_front_node(&mut self, mut new_head: Box<Node<T>>) {
        match self.list_head {
            None => {
                self.list_head = link_no_prev(new_head);
                self.list_tail = Rawlink::from(&mut self.list_head);
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
    fn pop_front_node(&mut self) -> Link<T> {
        self.list_head.take().map(|mut front_node| {
            self.length -= 1;
            match front_node.next.take() {
                Some(node) => self.list_head = link_no_prev(node),
                None => self.list_tail = Rawlink::none(),
            }
            front_node
        })
    }

    /// Add a Node last in the list
    #[inline]
    fn push_back_node(&mut self, new_tail: Box<Node<T>>) {
        match unsafe { self.list_tail.resolve_mut() } {
            None => return self.push_front_node(new_tail),
            Some(tail) => {
                tail.set_next(new_tail);
                self.list_tail = Rawlink::from(&mut tail.next);
            }
        }
        self.length += 1;
    }

    /// Remove the last Node and return it, or None if the list is empty
    #[inline]
    fn pop_back_node(&mut self) -> Link<T> {
        unsafe {
            self.list_tail.resolve_mut().and_then(|tail| {
                self.length -= 1;
                self.list_tail = tail.prev;
                match tail.prev.resolve_mut() {
                    None => self.list_head.take(),
                    Some(tail_prev) => tail_prev.next.take(),
                }
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for LinkedList<T> {
    #[inline]
    fn default() -> Self {
        LinkedList::new()
    }
}

impl<T> LinkedList<T> {
    /// Creates an empty `LinkedList`.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Self {
        LinkedList {
            list_head: None,
            list_tail: Rawlink::none(),
            length: 0,
        }
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
    /// for e in &a {
    ///     println!("{}", e); // prints 1, then 2, then 3, then 4
    /// }
    /// println!("{}", b.len()); // prints 0
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn append(&mut self, other: &mut LinkedList<T>) {
        match unsafe { self.list_tail.resolve_mut() } {
            None => {
                self.length = other.length;
                self.list_head = other.list_head.take();
                self.list_tail = other.list_tail.take();
            }
            Some(tail) => {
                // Carefully empty `other`.
                let o_tail = other.list_tail.take();
                let o_length = other.length;
                match other.list_head.take() {
                    None => return,
                    Some(node) => {
                        tail.set_next(node);
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
        Iter {
            nelem: self.len(),
            head: &self.list_head,
            tail: self.list_tail,
        }
    }

    /// Provides a forward iterator with mutable references.
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            nelem: self.len(),
            head: Rawlink::from(&mut self.list_head),
            tail: self.list_tail,
            list: self,
        }
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
        unsafe { self.list_tail.resolve().map(|tail| &tail.value) }
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
        unsafe { self.list_tail.resolve_mut().map(|tail| &mut tail.value) }
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
        self.pop_front_node().map(|box Node { value, .. }| value)
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
        self.pop_back_node().map(|box Node { value, .. }| value)
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
        } else {
            // better off starting from the end
            let mut iter = self.iter_mut();
            for _ in 0..len - 1 - (at - 1) {
                iter.next_back();
            }
            iter.tail
        };

        // The split node is the new tail node of the first part and owns
        // the head of the second part.
        let mut second_part_head;

        unsafe {
            second_part_head = split_node.resolve_mut().unwrap().next.take();
            match second_part_head {
                None => {}
                Some(ref mut head) => head.prev = Rawlink::none(),
            }
        }

        let second_part = LinkedList {
            list_head: second_part_head,
            list_tail: self.list_tail,
            length: len - at,
        };

        // Fix the tail ptr of the first part
        self.list_tail = split_node;
        self.length = at;

        second_part
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Drop for LinkedList<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        // Dissolve the linked_list in a loop.
        // Just dropping the list_head can lead to stack exhaustion
        // when length is >> 1_000_000
        while let Some(mut head_) = self.list_head.take() {
            self.list_head = head_.next.take();
        }
        self.length = 0;
        self.list_tail = Rawlink::none();
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for Iter<'a, A> {
    type Item = &'a A;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
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
        unsafe {
            self.tail.resolve().map(|prev| {
                self.nelem -= 1;
                self.tail = prev.prev;
                &prev.value
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> ExactSizeIterator for Iter<'a, A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, A> Iterator for IterMut<'a, A> {
    type Item = &'a mut A;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.nelem == 0 {
            return None;
        }
        unsafe {
            self.head.resolve_mut().map(|next| {
                self.nelem -= 1;
                self.head = Rawlink::from(&mut next.next);
                &mut next.value
            })
        }
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
        unsafe {
            self.tail.resolve_mut().map(|prev| {
                self.nelem -= 1;
                self.tail = prev.prev;
                &mut prev.value
            })
        }
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
        match unsafe { self.head.resolve_mut() } {
            None => {
                self.list.push_back_node(ins_node);
            }
            Some(node) => {
                let prev_node = match unsafe { node.prev.resolve_mut() } {
                    None => return self.list.push_front_node(ins_node),
                    Some(prev) => prev,
                };
                let node_own = prev_node.next.take().unwrap();
                ins_node.set_next(node_own);
                prev_node.set_next(ins_node);
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
    /// #![feature(linked_list_extras)]
    ///
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
    ///     assert_eq!(vec, [1, 2, 3, 4]);
    /// }
    /// ```
    #[inline]
    #[unstable(feature = "linked_list_extras",
               reason = "this is probably better handled by a cursor type -- we'll see",
               issue = "27794")]
    pub fn insert_next(&mut self, elt: A) {
        self.insert_next_node(box Node::new(elt))
    }

    /// Provides a reference to the next element, without changing the iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(linked_list_extras)]
    ///
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
    #[unstable(feature = "linked_list_extras",
               reason = "this is probably better handled by a cursor type -- we'll see",
               issue = "27794")]
    pub fn peek_next(&mut self) -> Option<&mut A> {
        if self.nelem == 0 {
            return None;
        }
        unsafe { self.head.resolve_mut().map(|head| &mut head.value) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Iterator for IntoIter<A> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.list.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.length, Some(self.list.length))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A> {
        self.list.pop_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> ExactSizeIterator for IntoIter<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> FromIterator<A> for LinkedList<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut ret = LinkedList::new();
        ret.extend(iter);
        ret
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    /// Consumes the list into an iterator yielding elements by value.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter { list: self }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a LinkedList<T> {
    type Item = &'a T;
    type IntoIter = Iter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(mut self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A> Extend<A> for LinkedList<A> {
    fn extend<T: IntoIterator<Item = A>>(&mut self, iter: T) {
        for elt in iter {
            self.push_back(elt);
        }
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy> Extend<&'a T> for LinkedList<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialEq> PartialEq for LinkedList<A> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }

    fn ne(&self, other: &LinkedList<A>) -> bool {
        self.len() != other.len() || self.iter().ne(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Eq> Eq for LinkedList<A> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: PartialOrd> PartialOrd for LinkedList<A> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Ord> Ord for LinkedList<A> {
    #[inline]
    fn cmp(&self, other: &LinkedList<A>) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Clone> Clone for LinkedList<A> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: fmt::Debug> fmt::Debug for LinkedList<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A: Hash> Hash for LinkedList<A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for elt in self {
            elt.hash(state);
        }
    }
}

// Ensure that `LinkedList` and its read-only iterators are covariant in their type parameters.
#[allow(dead_code)]
fn assert_covariance() {
    fn a<'a>(x: LinkedList<&'static str>) -> LinkedList<&'a str> { x }
    fn b<'i, 'a>(x: Iter<'i, &'static str>) -> Iter<'i, &'a str> { x }
    fn c<'a>(x: IntoIter<&'static str>) -> IntoIter<&'a str> { x }
}

#[cfg(test)]
mod tests {
    use std::clone::Clone;
    use std::iter::{Iterator, IntoIterator, Extend};
    use std::option::Option::{self, Some, None};
    use std::__rand::{thread_rng, Rng};
    use std::thread;
    use std::vec::Vec;

    use super::{LinkedList, Node};

    #[cfg(test)]
    fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
        v.iter().cloned().collect()
    }

    pub fn check_links<T>(list: &LinkedList<T>) {
        let mut len = 0;
        let mut last_ptr: Option<&Node<T>> = None;
        let mut node_ptr: &Node<T>;
        match list.list_head {
            None => {
                assert_eq!(0, list.length);
                return;
            }
            Some(ref node) => node_ptr = &**node,
        }
        loop {
            match unsafe { (last_ptr, node_ptr.prev.resolve()) } {
                (None, None) => {}
                (None, _) => panic!("prev link for list_head"),
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
        let v = vec![1, 2, 3, 4, 5];
        let u = vec![9, 8, 1, 2, 3, 4, 5];
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
    fn test_insert_prev() {
        let mut m = list_from(&[0, 2, 4, 6, 8]);
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
        assert_eq!(m.into_iter().collect::<Vec<_>>(),
                   [-2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]);
    }

    #[test]
    fn test_send() {
        let n = list_from(&[1, 2, 3]);
        thread::spawn(move || {
            check_links(&n);
            let a: &[_] = &[&1, &2, &3];
            assert_eq!(a, &n.iter().collect::<Vec<_>>()[..]);
        })
            .join()
            .ok()
            .unwrap();
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
    fn test_26021() {
        use std::iter::ExactSizeIterator;
        // There was a bug in split_off that failed to null out the RHS's head's prev ptr.
        // This caused the RHS's dtor to walk up into the LHS at drop and delete all of
        // its nodes.
        //
        // https://github.com/rust-lang/rust/issues/26021
        let mut v1 = LinkedList::new();
        v1.push_front(1u8);
        v1.push_front(1u8);
        v1.push_front(1u8);
        v1.push_front(1u8);
        let _ = v1.split_off(3); // Dropping this now should not cause laundry consumption
        assert_eq!(v1.len(), 3);

        assert_eq!(v1.iter().len(), 3);
        assert_eq!(v1.iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn test_split_off() {
        let mut v1 = LinkedList::new();
        v1.push_front(1u8);
        v1.push_front(1u8);
        v1.push_front(1u8);
        v1.push_front(1u8);

        // test all splits
        for ix in 0..1 + v1.len() {
            let mut a = v1.clone();
            let b = a.split_off(ix);
            check_links(&a);
            check_links(&b);
            a.extend(b);
            assert_eq!(v1, a);
        }
    }


    #[cfg(test)]
    fn fuzz_test(sz: i32) {
        let mut m: LinkedList<_> = LinkedList::new();
        let mut v = vec![];
        for i in 0..sz {
            check_links(&m);
            let r: u8 = thread_rng().next_u32() as u8;
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
                2 | 4 => {
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
        for (a, &b) in m.into_iter().zip(&v) {
            i += 1;
            assert_eq!(a, b);
        }
        assert_eq!(i, v.len());
    }
}
