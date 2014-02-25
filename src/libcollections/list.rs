// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A standard, garbage-collected linked list.

use std::container::Container;

#[deriving(Clone, Eq)]
#[allow(missing_doc)]
pub enum List<T> {
    Cons(T, @List<T>),
    Nil,
}

pub struct Items<'a, T> {
    priv head: &'a List<T>,
    priv next: Option<&'a @List<T>>
}

impl<'a, T> Iterator<&'a T> for Items<'a, T> {
    fn next(&mut self) -> Option<&'a T> {
        match self.next {
            None => match *self.head {
                Nil => None,
                Cons(ref value, ref tail) => {
                    self.next = Some(tail);
                    Some(value)
                }
            },
            Some(next) => match **next {
                Nil => None,
                Cons(ref value, ref tail) => {
                    self.next = Some(tail);
                    Some(value)
                }
            }
        }
    }
}

impl<T> List<T> {
    /// Returns a forward iterator
    pub fn iter<'a>(&'a self) -> Items<'a, T> {
        Items {
            head: self,
            next: None
        }
    }

    /// Returns the first element of a list
    pub fn head<'a>(&'a self) -> Option<&'a T> {
        match *self {
          Nil => None,
          Cons(ref head, _) => Some(head)
        }
    }

    /// Returns all but the first element of a list
    pub fn tail(&self) -> Option<@List<T>> {
        match *self {
            Nil => None,
            Cons(_, tail) => Some(tail)
        }
    }
}

impl<T> Container for List<T> {
    /// Returns the length of a list
    fn len(&self) -> uint { self.iter().len() }

    /// Returns true if the list is empty
    fn is_empty(&self) -> bool { match *self { Nil => true, _ => false } }
}

impl<T:Eq> List<T> {
    /// Returns true if a list contains an element with the given value
    pub fn contains(&self, element: T) -> bool {
        self.iter().any(|list_element| *list_element == element)
    }
}

impl<T:'static + Clone> List<T> {
    /// Create a list from a vector
    pub fn from_vec(v: &[T]) -> List<T> {
        match v.len() {
            0 => Nil,
            _ => v.rev_iter().fold(Nil, |tail, value: &T| Cons(value.clone(), @tail))
        }
    }

    /// Appends one list to another, returning a new list
    pub fn append(&self, other: List<T>) -> List<T> {
        match other {
            Nil => return self.clone(),
            _ => match *self {
                Nil => return other,
                Cons(ref value, tail) => Cons(value.clone(), @tail.append(other))
            }
        }
    }

    /// Push one element into the front of a list, returning a new list
    pub fn unshift(&self, element: T) -> List<T> {
        Cons(element, @(self.clone()))
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
        assert_eq!(empty, Nil::<int>);
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
        assert!((!empty.contains(5)));

        let list = List::from_vec([5, 8, 6]);
        assert!((list.contains(5)));
        assert!((!list.contains(7)));
        assert!((list.contains(8)));
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
        assert_eq!(List::from_vec([1, 2, 3, 4]),
                   List::from_vec([1, 2]).append(List::from_vec([3, 4])));
    }

    #[test]
    fn test_unshift() {
        let list = List::from_vec([1]);
        let new_list = list.unshift(0);
        assert_eq!(list.len(), 1u);
        assert_eq!(new_list.len(), 2u);
        assert_eq!(new_list, List::from_vec([0, 1]));
    }
}
