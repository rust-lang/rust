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

use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hasher, Hash};
use core::iter::{FromIterator, FusedIterator};
use core::marker::PhantomData;
use core::mem;
use core::ptr::NonNull;

use crate::boxed::Box;
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
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<Box<Node<T>>>,
}

struct Node<T> {
    next: Option<NonNull<Node<T>>>,
    prev: Option<NonNull<Node<T>>>,
    element: T,
}

/// An iterator over the elements of a `LinkedList`.
///
/// This `struct` is created by the [`iter`] method on [`LinkedList`]. See its
/// documentation for more.
///
/// [`iter`]: struct.LinkedList.html#method.iter
/// [`LinkedList`]: struct.LinkedList.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Iter<'a, T: 'a> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a Node<T>>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for Iter<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Iter")
         .field(&self.len)
         .finish()
    }
}

// FIXME(#26925) Remove in favor of `#[derive(Clone)]`
#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Clone for Iter<'_, T> {
    fn clone(&self) -> Self {
        Iter { ..*self }
    }
}

/// A mutable iterator over the elements of a `LinkedList`.
///
/// This `struct` is created by the [`iter_mut`] method on [`LinkedList`]. See its
/// documentation for more.
///
/// [`iter_mut`]: struct.LinkedList.html#method.iter_mut
/// [`LinkedList`]: struct.LinkedList.html
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IterMut<'a, T: 'a> {
    // We do *not* exclusively own the entire list here, references to node's `element`
    // have been handed out by the iterator!  So be careful when using this; the methods
    // called must be aware that there can be aliasing pointers to `element`.
    list: &'a mut LinkedList<T>,
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for IterMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IterMut")
         .field(&self.list)
         .field(&self.len)
         .finish()
    }
}

/// An owning iterator over the elements of a `LinkedList`.
///
/// This `struct` is created by the [`into_iter`] method on [`LinkedList`][`LinkedList`]
/// (provided by the `IntoIterator` trait). See its documentation for more.
///
/// [`into_iter`]: struct.LinkedList.html#method.into_iter
/// [`LinkedList`]: struct.LinkedList.html
#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct IntoIter<T> {
    list: LinkedList<T>,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl<T: fmt::Debug> fmt::Debug for IntoIter<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter")
         .field(&self.list)
         .finish()
    }
}

impl<T> Node<T> {
    fn new(element: T) -> Self {
        Node {
            next: None,
            prev: None,
            element,
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
        // This method takes care not to create mutable references to whole nodes,
        // to maintain validity of aliasing pointers into `element`.
        unsafe {
            node.next = self.head;
            node.prev = None;
            let node = Some(Box::into_raw_non_null(node));

            match self.head {
                None => self.tail = node,
                // Not creating new mutable (unique!) references overlapping `element`.
                Some(head) => (*head.as_ptr()).prev = node,
            }

            self.head = node;
            self.len += 1;
        }
    }

    /// Removes and returns the node at the front of the list.
    #[inline]
    fn pop_front_node(&mut self) -> Option<Box<Node<T>>> {
        // This method takes care not to create mutable references to whole nodes,
        // to maintain validity of aliasing pointers into `element`.
        self.head.map(|node| unsafe {
            let node = Box::from_raw(node.as_ptr());
            self.head = node.next;

            match self.head {
                None => self.tail = None,
                // Not creating new mutable (unique!) references overlapping `element`.
                Some(head) => (*head.as_ptr()).prev = None,
            }

            self.len -= 1;
            node
        })
    }

    /// Adds the given node to the back of the list.
    #[inline]
    fn push_back_node(&mut self, mut node: Box<Node<T>>) {
        // This method takes care not to create mutable references to whole nodes,
        // to maintain validity of aliasing pointers into `element`.
        unsafe {
            node.next = None;
            node.prev = self.tail;
            let node = Some(Box::into_raw_non_null(node));

            match self.tail {
                None => self.head = node,
                // Not creating new mutable (unique!) references overlapping `element`.
                Some(tail) => (*tail.as_ptr()).next = node,
            }

            self.tail = node;
            self.len += 1;
        }
    }

    /// Removes and returns the node at the back of the list.
    #[inline]
    fn pop_back_node(&mut self) -> Option<Box<Node<T>>> {
        // This method takes care not to create mutable references to whole nodes,
        // to maintain validity of aliasing pointers into `element`.
        self.tail.map(|node| unsafe {
            let node = Box::from_raw(node.as_ptr());
            self.tail = node.prev;

            match self.tail {
                None => self.head = None,
                // Not creating new mutable (unique!) references overlapping `element`.
                Some(tail) => (*tail.as_ptr()).next = None,
            }

            self.len -= 1;
            node
        })
    }

    /// Unlinks the specified node from the current list.
    ///
    /// Warning: this will not check that the provided node belongs to the current list.
    ///
    /// This method takes care not to create mutable references to `element`, to
    /// maintain validity of aliasing pointers.
    #[inline]
    unsafe fn unlink_node(&mut self, mut node: NonNull<Node<T>>) {
        let node = node.as_mut(); // this one is ours now, we can create an &mut.

        // Not creating new mutable (unique!) references overlapping `element`.
        match node.prev {
            Some(prev) => (*prev.as_ptr()).next = node.next.clone(),
            // this node is the head node
            None => self.head = node.next.clone(),
        };

        match node.next {
            Some(next) => (*next.as_ptr()).prev = node.prev.clone(),
            // this node is the tail node
            None => self.tail = node.prev.clone(),
        };

        self.len -= 1;
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
            Some(mut tail) => {
                // `as_mut` is okay here because we have exclusive access to the entirety
                // of both lists.
                if let Some(mut other_head) = other.head.take() {
                    unsafe {
                        tail.as_mut().next = Some(other_head);
                        other_head.as_mut().prev = Some(tail);
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
    pub fn iter(&self) -> Iter<'_, T> {
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
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
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
        unsafe {
            self.head.as_ref().map(|node| &node.as_ref().element)
        }
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
        unsafe {
            self.head.as_mut().map(|node| &mut node.as_mut().element)
        }
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
        unsafe {
            self.tail.as_ref().map(|node| &node.as_ref().element)
        }
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
        unsafe {
            self.tail.as_mut().map(|node| &mut node.as_mut().element)
        }
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

    /// Appends an element to the back of a list.
    ///
    /// This operation should compute in O(1) time.
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
    /// This operation should compute in O(1) time.
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
    /// This operation should compute in O(n) time.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
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
            return mem::take(self);
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
            second_part_head = split_node.unwrap().as_mut().next.take();
            if let Some(mut head) = second_part_head {
                head.as_mut().prev = None;
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

    /// Creates an iterator which uses a closure to determine if an element should be removed.
    ///
    /// If the closure returns true, then the element is removed and yielded.
    /// If the closure returns false, the element will remain in the list and will not be yielded
    /// by the iterator.
    ///
    /// Note that `drain_filter` lets you mutate every element in the filter closure, regardless of
    /// whether you choose to keep or remove it.
    ///
    /// # Examples
    ///
    /// Splitting a list into evens and odds, reusing the original list:
    ///
    /// ```
    /// #![feature(drain_filter)]
    /// use std::collections::LinkedList;
    ///
    /// let mut numbers: LinkedList<u32> = LinkedList::new();
    /// numbers.extend(&[1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 14, 15]);
    ///
    /// let evens = numbers.drain_filter(|x| *x % 2 == 0).collect::<LinkedList<_>>();
    /// let odds = numbers;
    ///
    /// assert_eq!(evens.into_iter().collect::<Vec<_>>(), vec![2, 4, 6, 8, 14]);
    /// assert_eq!(odds.into_iter().collect::<Vec<_>>(), vec![1, 3, 5, 9, 11, 13, 15]);
    /// ```
    #[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
    pub fn drain_filter<F>(&mut self, filter: F) -> DrainFilter<'_, T, F>
        where F: FnMut(&mut T) -> bool
    {
        // avoid borrow issues.
        let it = self.head;
        let old_len = self.len;

        DrainFilter {
            list: self,
            it: it,
            pred: filter,
            idx: 0,
            old_len: old_len,
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
                // Need an unbound lifetime to get 'a
                let node = &*node.as_ptr();
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

    #[inline]
    fn last(mut self) -> Option<&'a T> {
        self.next_back()
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
                // Need an unbound lifetime to get 'a
                let node = &*node.as_ptr();
                self.len -= 1;
                self.tail = node.prev;
                &node.element
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for Iter<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Iter<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe {
                // Need an unbound lifetime to get 'a
                let node = &mut *node.as_ptr();
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

    #[inline]
    fn last(mut self) -> Option<&'a mut T> {
        self.next_back()
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
                // Need an unbound lifetime to get 'a
                let node = &mut *node.as_ptr();
                self.len -= 1;
                self.tail = node.prev;
                &mut node.element
            })
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> ExactSizeIterator for IterMut<'_, T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for IterMut<'_, T> {}

impl<T> IterMut<'_, T> {
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
            // `push_back` is okay with aliasing `element` references
            None => self.list.push_back(element),
            Some(head) => unsafe {
                let prev = match head.as_ref().prev {
                    // `push_front` is okay with aliasing nodes
                    None => return self.list.push_front(element),
                    Some(prev) => prev,
                };

                let node = Some(Box::into_raw_non_null(box Node {
                    next: Some(head),
                    prev: Some(prev),
                    element,
                }));

                // Not creating references to entire nodes to not invalidate the
                // reference to `element` we handed to the user.
                (*prev.as_ptr()).next = node;
                (*head.as_ptr()).prev = node;

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
            unsafe {
                self.head.as_mut().map(|node| &mut node.as_mut().element)
            }
        }
    }
}

/// An iterator produced by calling `drain_filter` on LinkedList.
#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
pub struct DrainFilter<'a, T: 'a, F: 'a>
    where F: FnMut(&mut T) -> bool,
{
    list: &'a mut LinkedList<T>,
    it: Option<NonNull<Node<T>>>,
    pred: F,
    idx: usize,
    old_len: usize,
}

#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
impl<T, F> Iterator for DrainFilter<'_, T, F>
    where F: FnMut(&mut T) -> bool,
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        while let Some(mut node) = self.it {
            unsafe {
                self.it = node.as_ref().next;
                self.idx += 1;

                if (self.pred)(&mut node.as_mut().element) {
                    // `unlink_node` is okay with aliasing `element` references.
                    self.list.unlink_node(node);
                    return Some(Box::from_raw(node.as_ptr()).element);
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.old_len - self.idx))
    }
}

#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
impl<T, F> Drop for DrainFilter<'_, T, F>
    where F: FnMut(&mut T) -> bool,
{
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

#[unstable(feature = "drain_filter", reason = "recently added", issue = "43244")]
impl<T: fmt::Debug, F> fmt::Debug for DrainFilter<'_, T, F>
    where F: FnMut(&mut T) -> bool
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("DrainFilter")
         .field(&self.list)
         .finish()
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

#[stable(feature = "fused", since = "1.26.0")]
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
        iter.into_iter().for_each(move |elt| self.push_back(elt));
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
unsafe impl<T: Sync> Send for Iter<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for Iter<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Send> Send for IterMut<'_, T> {}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<T: Sync> Sync for IterMut<'_, T> {}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::vec::Vec;

    use rand::{thread_rng, RngCore};

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
                    // tail node should also be None.
                    assert!(list.tail.is_none());
                    assert_eq!(0, list.len);
                    return;
                }
                Some(node) => node_ptr = &*node.as_ptr(),
            }
            loop {
                match (last_ptr, node_ptr.prev) {
                    (None, None) => {}
                    (None, _) => panic!("prev link for head"),
                    (Some(p), Some(pptr)) => {
                        assert_eq!(p as *const Node<T>, pptr.as_ptr() as *const Node<T>);
                    }
                    _ => panic!("prev link is none, not good"),
                }
                match node_ptr.next {
                    Some(next) => {
                        last_ptr = Some(node_ptr);
                        node_ptr = &*next.as_ptr();
                        len += 1;
                    }
                    None => {
                        len += 1;
                        break;
                    }
                }
            }

            // verify that the tail node points to the last node.
            let tail = list.tail.as_ref().expect("some tail node").as_ref();
            assert_eq!(tail as *const Node<T>, node_ptr as *const Node<T>);
            // check that len matches interior links.
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
    #[cfg(not(miri))] // Miri does not support threads
    fn test_send() {
        let n = list_from(&[1, 2, 3]);
        thread::spawn(move || {
                check_links(&n);
                let a: &[_] = &[&1, &2, &3];
                assert_eq!(a, &*n.iter().collect::<Vec<_>>());
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
            #[cfg(not(miri))] // Miri is too slow
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

    #[test]
    fn drain_filter_test() {
        let mut m: LinkedList<u32> = LinkedList::new();
        m.extend(&[1, 2, 3, 4, 5, 6]);
        let deleted = m.drain_filter(|v| *v < 4).collect::<Vec<_>>();

        check_links(&m);

        assert_eq!(deleted, &[1, 2, 3]);
        assert_eq!(m.into_iter().collect::<Vec<_>>(), &[4, 5, 6]);
    }

    #[test]
    fn drain_to_empty_test() {
        let mut m: LinkedList<u32> = LinkedList::new();
        m.extend(&[1, 2, 3, 4, 5, 6]);
        let deleted = m.drain_filter(|_| true).collect::<Vec<_>>();

        check_links(&m);

        assert_eq!(deleted, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(m.into_iter().collect::<Vec<_>>(), &[]);
    }
}
