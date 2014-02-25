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
}

impl<T> Container for List<T> {
    /// Returns the length of a list
    fn len(&self) -> uint { self.iter().len() }

    /// Returns true if the list is empty
    fn is_empty(&self) -> bool { match *self { Nil => true, _ => false } }
}

/// Returns true if a list contains an element with the given value
pub fn has<T:Eq>(list: @List<T>, element: T) -> bool {
    let mut found = false;
    each(list, |e| {
        if *e == element { found = true; false } else { true }
    });
    return found;
}

/// Returns all but the first element of a list
pub fn tail<T>(list: @List<T>) -> @List<T> {
    match *list {
        Cons(_, tail) => return tail,
        Nil => fail!("list empty")
    }
}

/// Returns the first element of a list
pub fn head<T:Clone>(list: @List<T>) -> T {
    match *list {
      Cons(ref head, _) => (*head).clone(),
      // makes me sad
      _ => fail!("head invoked on empty list")
    }
}

/// Appends one list to another
pub fn append<T:Clone + 'static>(list: @List<T>, other: @List<T>) -> @List<T> {
    match *list {
      Nil => return other,
      Cons(ref head, tail) => {
        let rest = append(tail, other);
        return @Cons((*head).clone(), rest);
      }
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
}

/*
/// Push one element into the front of a list, returning a new list
/// THIS VERSION DOESN'T ACTUALLY WORK
fn push<T:Clone>(ll: &mut @list<T>, vv: T) {
    ll = &mut @cons(vv, *ll)
}
*/

#[cfg(test)]
mod tests {
    use list::{List, Nil, head, tail};
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
        let list = @List::from_vec([0, 1, 2]);

        assert_eq!(head(list), 0);

        let tail_l = tail(list);
        assert_eq!(head(tail_l), 1);

        let tail_tail_l = tail(tail_l);
        assert_eq!(head(tail_tail_l), 2);
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
    fn test_has() {
        let list = @List::from_vec([5, 8, 6]);
        let empty = @list::Nil::<int>;
        assert!((list::has(list, 5)));
        assert!((!list::has(list, 7)));
        assert!((list::has(list, 8)));
        assert!((!list::has(empty, 5)));
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
        assert!(@List::from_vec([1,2,3,4])
            == list::append(@List::from_vec([1,2]), @List::from_vec([3,4])));
    }
}
