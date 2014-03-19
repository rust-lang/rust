// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::container::Container;
use std::fmt;
use std::mem;

/// A linked list implementation.
///
/// # Example
///
/// ```rust
/// use collections::list::{List, Cons, Nil};
///
/// let mut list = List::new();
/// list.push(3);
/// list.push(2);
/// list.push(1);
/// assert_eq!(list, Cons(1, ~Cons(2, ~Cons(3, ~Nil))));
/// ```
pub enum List<T> {
    /// A node with a value in the linked list.
    Cons(T, ~List<T>),

    /// No value.
    Nil,
}

impl<T> List<T> {
    /// Create an empty linked list.
    ///
    /// ```rust
    /// use collections::list::{List, Nil};
    ///
    /// let list: List<int> = List::new();
    /// assert_eq!(list, Nil);
    /// ```
    #[inline]
    pub fn new() -> List<T> {
        Nil
    }

    /// Pushes a value at the front of the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.head(), None);
    ///
    /// list.push(5);
    /// assert_eq!(list.head().unwrap(), &5);
    /// ```
    #[inline]
    pub fn push(&mut self, value: T) {
        *self = Cons(value, ~mem::replace(self, Nil));
    }

    /// Removes the first element of a list, or `None` if empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.pop(), None);
    ///
    /// list.push(5);
    /// assert_eq!(list.pop(), Some(5));
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        match mem::replace(self, Nil) {
            Cons(value, ~tail) => {
                *self = tail;
                Some(value)
            }
            Nil => None,
        }
    }

    /// Provide a forward iterator.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let list = List::from_vec([1, 2, 3]);
    /// let mut iter = list.iter();
    /// assert_eq!(iter.next().unwrap(), &1);
    /// assert_eq!(iter.next().unwrap(), &2);
    /// assert_eq!(iter.next().unwrap(), &3);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items {
            head: self,
            next: None
        }
    }

    /// Provide a forward iterator that moves elements out of the list.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let list = List::from_vec([1, 2, 3]);
    /// let mut iter = list.move_iter();
    /// assert_eq!(iter.next().unwrap(), 1);
    /// assert_eq!(iter.next().unwrap(), 2);
    /// assert_eq!(iter.next().unwrap(), 3);
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        MoveItems {
            head: self,
        }
    }

    /// Returns the first element of a list.
    ///
    /// # example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.head(), None);
    ///
    /// list.push(1);
    /// assert_eq!(list.head().unwrap(), &1);
    #[inline]
    pub fn head<'a>(&'a self) -> Option<&'a T> {
        match *self {
          Nil => None,
          Cons(ref head, _) => Some(head)
        }
    }

    /// Returns all but the first element of a list.
    ///
    /// # example
    ///
    /// ```rust
    /// use collections::list::{List, Cons, Nil};
    ///
    /// let mut list = List::new();
    /// assert_eq!(list.tail(), None);
    ///
    /// list.push(1);
    /// assert_eq!(list.tail().unwrap(), &Nil);
    ///
    /// list.push(2);
    /// assert_eq!(list.tail().unwrap(), &Cons(1, ~Nil));
    #[inline]
    pub fn tail<'a>(&'a self) -> Option<&'a List<T>> {
        match *self {
            Nil => None,
            Cons(_, ref tail) => Some(&**tail)
        }
    }
}

impl<T: Clone> List<T> {
    /// Create a list from a vector.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::list::{List, Cons, Nil};
    ///
    /// let list = List::from_vec([1, 2, 3]);
    /// assert_eq!(list, Cons(1, ~Cons(2, ~Cons(3, ~Nil))));
    /// ```
    pub fn from_vec(v: &[T]) -> List<T> {
        match v.len() {
            0 => Nil,
            _ => {
                v.rev_iter().fold(Nil, |tail, value: &T| Cons(value.clone(), ~tail))
            }
        }
    }

    /// Creates a reversed list from an iterator. This is faster than using
    /// `FromIterator::from_iterator`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let v = ~[1, 2, 3];
    /// let list: List<int> = List::from_iterator_rev(&mut v.move_iter());
    /// assert_eq!(list, List::from_vec([3, 2, 1]));
    /// ```
    pub fn from_iterator_rev<Iter: Iterator<T>>(iterator: &mut Iter) -> List<T> {
        let mut list = List::new();

        for elt in *iterator {
            list.push(elt);
        }

        list
    }

    /// Appends one list at the end of another.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let a = List::from_vec([1, 2, 3]);
    /// let b = List::from_vec([4, 5, 6]);
    /// let c = a.append(b);
    /// assert_eq!(c, List::from_vec([1, 2, 3, 4, 5, 6]));
    /// ```
    pub fn append(self, other: List<T>) -> List<T> {
        match (self, other) {
            (Nil, other) => other,
            (self_, Nil) => self_,
            (self_, other) => {
                let mut list = List::from_iterator_rev(&mut self_.move_iter());

                for elt in other.move_iter() {
                    list.push(elt);
                }

                list.reverse();

                list
            }
        }
    }

    /// Reverses a list in place.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let mut list = List::from_vec([1, 2, 3]);
    /// list.reverse();
    /// assert_eq!(list, List::from_vec([3, 2, 1]));
    /// ```
    pub fn reverse(&mut self) {
        for elt in mem::replace(self, Nil).move_iter() {
            self.push(elt);
        }
    }
}

impl<T: Eq> List<T> {
    /// Return true if the list contains a value.
    ///
    /// # Example
    ///
    /// ```rust
    /// use collections::List;
    ///
    /// let list = List::from_vec([1, 2, 3]);
    /// assert!(list.contains(&2));
    /// assert!(!list.contains(&4));
    /// ```
    pub fn contains(&self, element: &T) -> bool {
        self.iter().any(|list_element| list_element == element)
    }
}

impl<T: Clone> Clone for List<T> {
    fn clone(&self) -> List<T> {
        // Contruct the list in reversed order to avoid a stack overflow.
        let mut list = List::new();

        for elt in self.iter() {
            list.push(elt.clone());
        }

        list.reverse();

        list
    }
}

impl<T: Eq> Eq for List<T> {
    fn eq(&self, other: &List<T>) -> bool {
        // Explicitly implement `Eq` to avoid running out of stack while
        // comparing lists.
        let mut list0 = self;
        let mut list1 = other;

        loop {
            match *list0 {
                Nil => {
                    match *list1 {
                        Nil => { return true; }
                        Cons(_, _) => { return false; }
                    }
                }
                Cons(ref v0, ref t0) => {
                    match *list1 {
                        Nil => { return false; }
                        Cons(ref v1, ref t1) => {
                            if v0 != v1 { return false; }

                            list0 = &**t0;
                            list1 = &**t1;
                        }
                    }
                }
            }
        }
    }
}

impl<T> Container for List<T> {
    fn len(&self) -> uint { self.iter().len() }

    fn is_empty(&self) -> bool { match *self { Nil => true, _ => false } }
}

impl<T: Clone> FromIterator<T> for List<T> {
    fn from_iterator<Iter: Iterator<T>>(iterator: &mut Iter) -> List<T> {
        let mut list = List::from_iterator_rev(iterator);
        list.reverse();
        list
    }
}

impl<T: Clone> Extendable<T> for List<T> {
    fn extend<Iter: Iterator<T>>(&mut self, iterator: &mut Iter) {
        let mut list = List::from_iterator_rev(iterator);

        for elt in self.iter() {
            list.push(elt.clone());
        }

        list.reverse();

        *self = list;
    }
}

impl<T: fmt::Show> fmt::Show for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f.buf, "["));
        let mut is_first = true;
        for x in self.iter() {
            if is_first {
                is_first = false;
            } else {
                try!(write!(f.buf, ", "));
            }
            try!(write!(f.buf, "{}", *x))
        }
        write!(f.buf, "]")
    }
}

/// A linked list iterator. See `List::iter` for a usage example.
pub struct Items<'a, T> {
    priv head: &'a List<T>,
    priv next: Option<&'a ~List<T>>
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        match self.next {
            None => {
                match *self.head {
                    Nil => None,
                    Cons(ref value, ref tail) => {
                        self.next = Some(tail);
                        Some(value)
                    }
                }
            }
            Some(next) => {
                match **next {
                    Nil => None,
                    Cons(ref value, ref tail) => {
                        self.next = Some(tail);
                        Some(value)
                    }
                }
            }
        }
    }
}

/// A linked list iterator that moves the elements out of the list. See
/// `List::move_iter` for a usage example.
pub struct MoveItems<T> {
    priv head: List<T>,
}

impl<T> Iterator<T> for MoveItems<T> {
    fn next(&mut self) -> Option<T> {
        match mem::replace(&mut self.head, Nil) {
            Nil => None,
            Cons(value, tail) => {
                self.head = *tail;
                Some(value)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use list::{List, Nil};
    use list;

    #[test]
    fn test_iter() {
        let list = List::from_vec([0, 1, 2]);
        let mut iter = list.iter();
        assert_eq!(&0, iter.next().unwrap());
        assert_eq!(&1, iter.next().unwrap());
        assert_eq!(&2, iter.next().unwrap());
        assert_eq!(None, iter.next());
    }

    #[test]
    fn test_is_empty() {
        let empty : list::List<int> = List::from_vec([]);
        let full1 = List::from_vec([1]);
        let full2 = List::from_vec(['r', 'u']);

        assert!(empty.is_empty());
        assert!(!full1.is_empty());
        assert!(!full2.is_empty());
    }

    #[test]
    fn test_from_vec() {
        let list = List::from_vec([0, 1, 2]);
        assert_eq!(list.head().unwrap(), &0);

        let mut tail = list.tail().unwrap();
        assert_eq!(tail.head().unwrap(), &1);

        tail = tail.tail().unwrap();
        assert_eq!(tail.head().unwrap(), &2);
    }

    #[test]
    fn test_from_vec_empty() {
        let empty : list::List<int> = List::from_vec([]);
        assert!(empty == Nil::<int>);
    }

    #[test]
    fn test_fold() {
        fn add_(a: uint, b: &uint) -> uint { a + *b }
        fn subtract_(a: uint, b: &uint) -> uint { a - *b }

        let empty = Nil::<uint>;
        assert_eq!(empty.iter().fold(0u, add_), 0u);
        assert_eq!(empty.iter().fold(10u, subtract_), 10u);

        let list = List::from_vec([0u, 1u, 2u, 3u, 4u]);
        assert_eq!(list.iter().fold(0u, add_), 10u);
        assert_eq!(list.iter().fold(10u, subtract_), 0u);
    }

    #[test]
    fn test_find_success() {
        fn match_(i: & &int) -> bool { **i == 2 }

        let list = List::from_vec([0, 1, 2]);
        assert_eq!(list.iter().find(match_).unwrap(), &2);
    }

    #[test]
    fn test_find_fail() {
        fn match_(_i: & &int) -> bool { false }

        let empty = Nil::<int>;
        assert_eq!(empty.iter().find(match_), None);

        let list = List::from_vec([0, 1, 2]);
        assert_eq!(list.iter().find(match_), None);
    }

    #[test]
    fn test_any() {
        fn match_(i: &int) -> bool { *i == 2 }

        let empty = Nil::<int>;
        assert_eq!(empty.iter().any(match_), false);

        let list = List::from_vec([0, 1, 2]);
        assert_eq!(list.iter().any(match_), true);
    }

    #[test]
    fn test_contains() {
        let empty = Nil::<int>;
        assert!((!empty.contains(&5)));

        let list = List::from_vec([5, 8, 6]);
        assert!((list.contains(&5)));
        assert!((!list.contains(&7)));
        assert!((list.contains(&8)));
    }

    #[test]
    fn test_len() {
        let empty = Nil::<int>;
        assert_eq!(empty.len(), 0u);

        let list = List::from_vec([0, 1, 2]);
        assert_eq!(list.len(), 3u);
    }

    #[test]
    fn test_append() {
        assert!(List::from_vec([1, 2, 3, 4]) ==
                List::from_vec([1, 2]).append(List::from_vec([3, 4])));
    }

    #[test]
    fn test_push() {
        let mut list = List::from_vec([1]);
        list.push(0);
        assert_eq!(list.len(), 2);
        assert!(list == List::from_vec([0, 1]));
    }
}
