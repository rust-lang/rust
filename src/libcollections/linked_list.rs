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
//! The `LinkedList` allows pushing and popping elements at either end
//! in constant time.
//!
//! Almost always it is better to use `Vec` or [`VecDeque`] instead of
//! [`LinkedList`]. In general, array-based containers are faster,
//! more memory efficient and make better use of CPU cache.
//!
//! [`LinkedList`]: ../linked_list/struct.LinkedList.html
//! [`VecDeque`]: ../vec_deque/struct.VecDeque.html

#![stable(feature = "rust1", since = "1.0.0")]

use alloc::boxed::{Box, IntermediateBox};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hasher, Hash};
use core::iter::{FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem;
use core::ops::{BoxPlace, InPlace, Place, Placer};
use core::ptr::{self, Shared};

use super::SpecExtend;

/// A doubly-linked list with owned nodes.
///
/// The `LinkedList` allows pushing and popping elements at either end
/// in constant time.
///
/// Almost always it is better to use `Vec` or `VecDeque` instead of
/// `LinkedList`. In general, array-based containers are faster,
/// more memory efficient and make better use of CPU cache.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct LinkedList<T> {
    head: Option<Shared<Node<T>>>,
    tail: Option<Shared<Node<T>>>,
    len: usize,
    marker: PhantomData<Box<Node<T>>>,
}

struct Node<T> {
    next: Option<Shared<Node<T>>>,
    prev: Option<Shared<Node<T>>>,
    element: T,
}

/// An iterator over references to the elements of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    head: Option<Shared<Node<T>>>,
    tail: Option<Shared<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a Node<T>>,
}

// FIXME #19839: deriving is too aggressive on the bounds (T doesn't need to be Clone).
#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Clone for Iter<'a, T> {
    fn clone(&self) -> Self {
        Iter { ..*self }
    }
}

/// An iterator over mutable references to the elements of a `LinkedList`.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    list: &'a mut LinkedList<T>,
    head: Option<Shared<Node<T>>>,
    tail: Option<Shared<Node<T>>>,
    len: usize,
}

/// An iterator over the elements of a `LinkedList`.
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    list: LinkedList<T>,
}

impl<T> Node<T> {
    fn new(element: T) -> Self {
        Node {
            next: None,
            prev: None,
            element: element,
        }
    }

    fn into_element(self: Box<Self>) -> T {
        self.element
    }
}

// private methods
impl<T> LinkedList<T> {
    /// Adds the given node to the front of the list.
    #[inline]
    fn push_front_node(&mut self, mut node: Box<Node<T>>) {
        unsafe {
            node.next = self.head;
            node.prev = None;
            let node = Some(Shared::new(Box::into_raw(node)));

            match self.head {
                None => self.tail = node,
                Some(head) => (**head).prev = node,
            }

            self.head = node;
            self.len += 1;
        }
    }

    /// Removes and returns the node at the front of the list.
    #[inline]
    fn pop_front_node(&mut self) -> Option<Box<Node<T>>> {
        self.head.map(|node| unsafe {
            let node = Box::from_raw(*node);
            self.head = node.next;

            match self.head {
                None => self.tail = None,
                Some(head) => (**head).prev = None,
            }

            self.len -= 1;
            node
        })
    }

    /// Adds the given node to the back of the list.
    #[inline]
    fn push_back_node(&mut self, mut node: Box<Node<T>>) {
        unsafe {
            node.next = None;
            node.prev = self.tail;
            let node = Some(Shared::new(Box::into_raw(node)));

            match self.tail {
                None => self.head = node,
                Some(tail) => (**tail).next = node,
            }

            self.tail = node;
            self.len += 1;
        }
    }

    /// Removes and returns the node at the back of the list.
    #[inline]
    fn pop_back_node(&mut self) -> Option<Box<Node<T>>> {
        self.tail.map(|node| unsafe {
            let node = Box::from_raw(*node);
            self.tail = node.prev;

            match self.tail {
                None => self.head = None,
                Some(tail) => (**tail).next = None,
            }

            self.len -= 1;
            node
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for LinkedList<T> {
    /// Creates an empty `LinkedList<T>`.
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LinkedList<T> {
    /// Creates an empty `LinkedList`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let list: LinkedList<u32> = LinkedList::new();
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Self {
        LinkedList {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
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
    /// let mut list1 = LinkedList::new();
    /// list1.push_back('a');
    ///
    /// let mut list2 = LinkedList::new();
    /// list2.push_back('b');
    /// list2.push_back('c');
    ///
    /// list1.append(&mut list2);
    ///
    /// let mut iter = list1.iter();
    /// assert_eq!(iter.next(), Some(&'a'));
    /// assert_eq!(iter.next(), Some(&'b'));
    /// assert_eq!(iter.next(), Some(&'c'));
    /// assert!(iter.next().is_none());
    ///
    /// assert!(list2.is_empty());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn append(&mut self, other: &mut Self) {
        match self.tail {
            None => mem::swap(self, other),
            Some(tail) => {
                if let Some(other_head) = other.head.take() {
                    unsafe {
                        (**tail).next = Some(other_head);
                        (**other_head).prev = Some(tail);
                    }

                    self.tail = other.tail.take();
                    self.len += mem::replace(&mut other.len, 0);
                }
            }
        }
    }

    /// Provides a forward iterator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut list: LinkedList<u32> = LinkedList::new();
    ///
    /// list.push_back(0);
    /// list.push_back(1);
    /// list.push_back(2);
    ///
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter(&self) -> Iter<T> {
        Iter {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }

    /// Provides a forward iterator with mutable references.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut list: LinkedList<u32> = LinkedList::new();
    ///
    /// list.push_back(0);
    /// list.push_back(1);
    /// list.push_back(2);
    ///
    /// for element in list.iter_mut() {
    ///     *element += 10;
    /// }
    ///
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next(), Some(&10));
    /// assert_eq!(iter.next(), Some(&11));
    /// assert_eq!(iter.next(), Some(&12));
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut {
            head: self.head,
            tail: self.tail,
            len: self.len,
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
        self.head.is_none()
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn len(&self) -> usize {
        self.len
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Returns `true` if the `LinkedList` contains an element equal to the
    /// given value.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::LinkedList;
    ///
    /// let mut list: LinkedList<u32> = LinkedList::new();
    ///
    /// list.push_back(0);
    /// list.push_back(1);
    /// list.push_back(2);
    ///
    /// assert_eq!(list.contains(&0), true);
    /// assert_eq!(list.contains(&10), false);
    /// ```
    #[stable(feature = "linked_list_contains", since = "1.12.0")]
    pub fn contains(&self, x: &T) -> bool
        where T: PartialEq<T>
    {
        self.iter().any(|e| e == x)
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front(&self) -> Option<&T> {
        self.head.map(|node| unsafe { &(**node).element })
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn front_mut(&mut self) -> Option<&mut T> {
        self.head.map(|node| unsafe { &mut (**node).element })
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back(&self) -> Option<&T> {
        self.tail.map(|node| unsafe { &(**node).element })
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
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn back_mut(&mut self) -> Option<&mut T> {
        self.tail.map(|node| unsafe { &mut (**node).element })
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn push_front(&mut self, elt: T) {
        self.push_front_node(box Node::new(elt));
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
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn pop_front(&mut self) -> Option<T> {
        self.pop_front_node().map(Node::into_element)
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
        self.push_back_node(box Node::new(elt));
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
        self.pop_back_node().map(Node::into_element)
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
            return mem::replace(self, Self::new());
        } else if at == len {
            return Self::new();
        }

        // Below, we iterate towards the `i-1`th node, either from the start or the end,
        // depending on which would be faster.
        let split_node = if at - 1 <= len - 1 - (at - 1) {
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
        let second_part_head;

        unsafe {
            second_part_head = (**split_node.unwrap()).next.take();
            if let Some(head) = second_part_head {
                (**head).prev = None;
            }
        }

        let second_part = LinkedList {
            head: second_part_head,
            tail: self.tail,
            len: len - at,
            marker: PhantomData,
        };

        // Fix the tail ptr of the first part
        self.tail = split_node;
        self.len = at;

        second_part
    }

    /// Returns a place for insertion at the front of the list.
    ///
    /// Using this method with placement syntax is equivalent to [`push_front`]
    /// (#method.push_front), but may be more efficient.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collection_placement)]
    /// #![feature(placement_in_syntax)]
    ///
    /// use std::collections::LinkedList;
    ///
    /// let mut list = LinkedList::new();
    /// list.front_place() <- 2;
    /// list.front_place() <- 4;
    /// assert!(list.iter().eq(&[4, 2]));
    /// ```
    #[unstable(feature = "collection_placement",
               reason = "method name and placement protocol are subject to change",
               issue = "30172")]
    pub fn front_place(&mut self) -> FrontPlace<T> {
        FrontPlace {
            list: self,
            node: IntermediateBox::make_place(),
        }
    }

    /// Returns a place for insertion at the back of the list.
    ///
    /// Using this method with placement syntax is equivalent to [`push_back`](#method.push_back),
    /// but may be more efficient.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(collection_placement)]
    /// #![feature(placement_in_syntax)]
    ///
    /// use std::collections::LinkedList;
    ///
    /// let mut list = LinkedList::new();
    /// list.back_place() <- 2;
    /// list.back_place() <- 4;
    /// assert!(list.iter().eq(&[2, 4]));
    /// ```
    #[unstable(feature = "collection_placement",
               reason = "method name and placement protocol are subject to change",
               issue = "30172")]
    pub fn back_place(&mut self) -> BackPlace<T> {
        BackPlace {
            list: self,
            node: IntermediateBox::make_place(),
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T> Drop for LinkedList<T> {
    fn drop(&mut self) {
        while let Some(_) = self.pop_front_node() {}
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe {
                let node = &**node;
                self.len -= 1;
                self.head = node.next;
                &node.element
            })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        if self.len == 0 {
            None
        } else {
            self.tail.map(|node| unsafe {
                let node = &**node;
                self.len -= 1;
                self.tail = node.prev;
                &node.element
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe {
                let node = &mut **node;
                self.len -= 1;
                self.head = node.next;
                &mut node.element
            })
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            None
        } else {
            self.tail.map(|node| unsafe {
                let node = &mut **node;
                self.len -= 1;
                self.tail = node.prev;
                &mut node.element
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<'a, T> FusedIterator for IterMut<'a, T> {}

impl<'a, T> IterMut<'a, T> {
    /// Inserts the given element just after the element most recently returned by `.next()`.
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
    pub fn insert_next(&mut self, element: T) {
        match self.head {
            None => self.list.push_back(element),
            Some(head) => unsafe {
                let prev = match (**head).prev {
                    None => return self.list.push_front(element),
                    Some(prev) => prev,
                };

                let node = Some(Shared::new(Box::into_raw(box Node {
                    next: Some(head),
                    prev: Some(prev),
                    element: element,
                })));

                (**prev).next = node;
                (**head).prev = node;

                self.list.len += 1;
            },
        }
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
    pub fn peek_next(&mut self) -> Option<&mut T> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe { &mut (**node).element })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Iterator for IntoIter<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.list.pop_front()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.list.len, Some(self.list.len))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> DoubleEndedIterator for IntoIter<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.list.pop_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IntoIter<T> {}

#[unstable(feature = "fused", issue = "35602")]
impl<T> FusedIterator for IntoIter<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> FromIterator<T> for LinkedList<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut list = Self::new();
        list.extend(iter);
        list
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> IntoIterator for LinkedList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    /// Consumes the list into an iterator yielding elements by value.
    #[inline]
    fn into_iter(self) -> IntoIter<T> {
        IntoIter { list: self }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> IntoIterator for &'a mut LinkedList<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        self.iter_mut()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Extend<T> for LinkedList<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        <Self as SpecExtend<I>>::spec_extend(self, iter);
    }
}

impl<I: IntoIterator> SpecExtend<I> for LinkedList<I::Item> {
    default fn spec_extend(&mut self, iter: I) {
        for elt in iter {
            self.push_back(elt);
        }
    }
}

impl<T> SpecExtend<LinkedList<T>> for LinkedList<T> {
    fn spec_extend(&mut self, ref mut other: LinkedList<T>) {
        self.append(other);
    }
}

#[stable(feature = "extend_ref", since = "1.2.0")]
impl<'a, T: 'a + Copy> Extend<&'a T> for LinkedList<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().cloned());
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialEq> PartialEq for LinkedList<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other)
    }

    fn ne(&self, other: &Self) -> bool {
        self.len() != other.len() || self.iter().ne(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Eq> Eq for LinkedList<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: PartialOrd> PartialOrd for LinkedList<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.iter().partial_cmp(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Ord> Ord for LinkedList<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for LinkedList<T> {
    fn clone(&self) -> Self {
        self.iter().cloned().collect()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug> fmt::Debug for LinkedList<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Hash> Hash for LinkedList<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.len().hash(state);
        for elt in self {
            elt.hash(state);
        }
    }
}

unsafe fn finalize<T>(node: IntermediateBox<Node<T>>) -> Box<Node<T>> {
    let mut node = node.finalize();
    ptr::write(&mut node.next, None);
    ptr::write(&mut node.prev, None);
    node
}

/// A place for insertion at the front of a `LinkedList`.
///
/// See [`LinkedList::front_place`](struct.LinkedList.html#method.front_place) for details.
#[must_use = "places do nothing unless written to with `<-` syntax"]
#[unstable(feature = "collection_placement",
           reason = "struct name and placement protocol are subject to change",
           issue = "30172")]
pub struct FrontPlace<'a, T: 'a> {
    list: &'a mut LinkedList<T>,
    node: IntermediateBox<Node<T>>,
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Placer<T> for FrontPlace<'a, T> {
    type Place = Self;

    fn make_place(self) -> Self {
        self
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Place<T> for FrontPlace<'a, T> {
    fn pointer(&mut self) -> *mut T {
        unsafe { &mut (*self.node.pointer()).element }
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> InPlace<T> for FrontPlace<'a, T> {
    type Owner = ();

    unsafe fn finalize(self) {
        let FrontPlace { list, node } = self;
        list.push_front_node(finalize(node));
    }
}

/// A place for insertion at the back of a `LinkedList`.
///
/// See [`LinkedList::back_place`](struct.LinkedList.html#method.back_place) for details.
#[must_use = "places do nothing unless written to with `<-` syntax"]
#[unstable(feature = "collection_placement",
           reason = "struct name and placement protocol are subject to change",
           issue = "30172")]
pub struct BackPlace<'a, T: 'a> {
    list: &'a mut LinkedList<T>,
    node: IntermediateBox<Node<T>>,
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Placer<T> for BackPlace<'a, T> {
    type Place = Self;

    fn make_place(self) -> Self {
        self
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> Place<T> for BackPlace<'a, T> {
    fn pointer(&mut self) -> *mut T {
        unsafe { &mut (*self.node.pointer()).element }
    }
}

#[unstable(feature = "collection_placement",
           reason = "placement protocol is subject to change",
           issue = "30172")]
impl<'a, T> InPlace<T> for BackPlace<'a, T> {
    type Owner = ();

    unsafe fn finalize(self) {
        let BackPlace { list, node } = self;
        list.push_back_node(finalize(node));
    }
}

// Ensure that `LinkedList` and its read-only iterators are covariant in their type parameters.
#[allow(dead_code)]
fn assert_covariance() {
    fn a<'a>(x: LinkedList<&'static str>) -> LinkedList<&'a str> {
        x
    }
    fn b<'i, 'a>(x: Iter<'i, &'static str>) -> Iter<'i, &'a str> {
        x
    }
    fn c<'a>(x: IntoIter<&'static str>) -> IntoIter<&'a str> {
        x
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for LinkedList<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for LinkedList<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Send for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Sync for Iter<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Send> Send for IterMut<'a, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<'a, T: Sync> Sync for IterMut<'a, T> {}

#[cfg(test)]
mod tests {
    use std::__rand::{thread_rng, Rng};
    use std::thread;
    use std::vec::Vec;

    use super::{LinkedList, Node};

    #[cfg(test)]
    fn list_from<T: Clone>(v: &[T]) -> LinkedList<T> {
        v.iter().cloned().collect()
    }

    pub fn check_links<T>(list: &LinkedList<T>) {
        unsafe {
            let mut len = 0;
            let mut last_ptr: Option<&Node<T>> = None;
            let mut node_ptr: &Node<T>;
            match list.head {
                None => {
                    assert_eq!(0, list.len);
                    return;
                }
                Some(node) => node_ptr = &**node,
            }
            loop {
                match (last_ptr, node_ptr.prev) {
                    (None, None) => {}
                    (None, _) => panic!("prev link for head"),
                    (Some(p), Some(pptr)) => {
                        assert_eq!(p as *const Node<T>, *pptr as *const Node<T>);
                    }
                    _ => panic!("prev link is none, not good"),
                }
                match node_ptr.next {
                    Some(next) => {
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
            assert_eq!(len, list.len);
        }
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
        sum.extend_from_slice(&u);
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
    #[cfg_attr(target_os = "emscripten", ignore)]
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
        // There was a bug in split_off that failed to null out the RHS's head's prev ptr.
        // This caused the RHS's dtor to walk up into the LHS at drop and delete all of
        // its nodes.
        //
        // https://github.com/rust-lang/rust/issues/26021
        let mut v1 = LinkedList::new();
        v1.push_front(1);
        v1.push_front(1);
        v1.push_front(1);
        v1.push_front(1);
        let _ = v1.split_off(3); // Dropping this now should not cause laundry consumption
        assert_eq!(v1.len(), 3);

        assert_eq!(v1.iter().len(), 3);
        assert_eq!(v1.iter().collect::<Vec<_>>().len(), 3);
    }

    #[test]
    fn test_split_off() {
        let mut v1 = LinkedList::new();
        v1.push_front(1);
        v1.push_front(1);
        v1.push_front(1);
        v1.push_front(1);

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
