// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*! 
 * This is a straight-forward implementation of a linked-list.
 *
 * This implementation also uses raw pointers, and no unsafe code.
 */

#[crate_id = "linked_list"];
#[crate_type="lib"];

use std::cmp::Eq;
use std::container::Container;
use std::util;
use std::iter::Iterator;
use std::ops::Index;

/// The basic implementation of a linked-list.
#[deriving(Eq, Clone)]
pub enum LinkedList<T> {
    Cell(T, ~LinkedList<T>),
    End
}

/// An iterator implementation for the linked-list.
/// Gives access to all the Iterator goodness (fold, map, ...) for free.
///
/// See the Iterator doc page for precise description of all the usable methods.
pub struct LinkedListIterator<'a, T> {
    current: &'a LinkedList<T>
}

/// A reference iterator implementation for the linked-list.
/// Gives access to all the Iterator goodness (fold, map, ...) for free.
///
/// See the Iterator doc page for precise description of all the usable methods.
pub struct LinkedListRefIterator<'a, T> {
    current: &'a LinkedList<T>
}

/// A moving iterator implementation for the linked-list.
/// Gives access to all the Iterator goodness (fold, map, ...) for free.
///
/// See the Iterator doc page for precise description of all the usable methods.
pub struct LinkedListMoveIterator<T> {
    current: ~LinkedList<T>
}

impl<T: Clone> Index<uint, T> for LinkedList<T>{
    /*!
     Allows the use of the [] operator on a list to access a copy of the element stored there

     #Note
     As the returned value is a copy of the original, you can't mutate the list using this accessor.
     This is due to a current issue in Rust (see issue 6515)

     If you want to replace a value in the list, use the provided replace method.
    */
    fn index(&self, i: &uint) -> T{
        match self{
            &Cell(ref x, ref xs) => {
                if *i == 0{
                    (*x).clone()
                } else {
                    xs.index(&(*i-1))
                }
            },
            &End                 => fail!("You can't index after the end of the list")
        }
    }
}

impl<T> Container for LinkedList<T> {
    /** 
     Returns the length of the list.

     ```rust
     use linked_list::LinkedList;
     let list = LinkedList::from_vec([1,2,3,4,5]);
     assert!(list.len() == 5);
     ```
     */
    fn len(&self) -> uint{
        match self{
            &End             => 0,
            &Cell(_, ref xs) => xs.len() + 1
        }
    }
}

impl<T: Eq+Clone> FromIterator<T> for ~LinkedList<T> {
    // Naive implementation which doesn't keep a ref to the last added cell
    // because the compiler doesn't seem to end the lifetime of the borrow when it should
    // (or I suck at dealing with lifetimes, which is also very probable)
    fn from_iterator<I: Iterator<T>>(iter: &mut I) -> ~LinkedList<T> {
        let mut list = LinkedList::<T>::new();
        let mut e = iter.next();

        // For some reason, I can't use "for e in iter"
        while e.is_some() {
            list.append(e.unwrap());
            e = iter.next();
        }
        list
    }
}

impl<T:Eq + Clone> LinkedList<T> {
    /**
     Allocates a new LinkedList and returns a ~pointer to it

     #Example
     ```rust
     use linked_list::{LinkedList, End};
     let list = LinkedList::<int>::new();
     assert!(list == ~End);
     ```
     */
    pub fn new() -> ~LinkedList<T>{
        ~End
    }

    /** 
     Creates a new linked list from a vector, and returns a ~pointer to it

     #Example
     ```rust
     use linked_list::{LinkedList, End, Cell};
     let list = LinkedList::from_vec([1,2,3]);
     assert!(list == ~Cell(1, ~Cell(2, ~Cell(3, ~End))));
     ```
    */
    pub fn from_vec(v: &[T]) -> ~LinkedList<T>{
        v.rev_iter().fold(~End::<T>, |tail, head| ~Cell((*head).clone(), tail))
    }

    /**
     Appends an element to the end on the list, and returns an &mut pointer to the list,
     which is useful to be able to chain append() calls together.

     #Example

     ```rust
     use linked_list::LinkedList;    
     let mut list = LinkedList::new();
     list.append(1).append(2).append(3);
     let expected_result = LinkedList::from_vec([1,2,3]);
     assert!(list == expected_result);
     ```
     
     #Note
     append() will NOT return the original list, only an &mut pointer to the last Cell.

     This is ok for chaining purposes, but might be a bit surprising/unintuitive.
     
     Because of this, you can't do

     ```
     use linked_list::LinkedList;
     let list = LinkedList::new().append(1).append(2).append(3);
     // list now has type &mut LinkedList, and is &mut Cell(3, ~End)
     ```
    */
    pub fn append<'a>(&'a mut self, obj: T) -> &'a mut LinkedList<T>{
        match self{
            &End                 => {*self = Cell(obj, ~End); self},
            &Cell(_, ref mut xs) => xs.append(obj)
        }
    }

    /**
     Prepends an element to the list, and returns an &mut pointer to the list
     for chaining purposes.

     #Example

     ```rust
     use linked_list::LinkedList;
     let mut list = LinkedList::new();
     list.prepend(5).prepend(4).prepend(3);
     assert!(list == LinkedList::from_vec([3,4,5]));
     ```

     #Note
     The returned pointer is an &pointer, which means if you use allocation & prepend you lose
     ownership :

     ```
     let list = LinkedList::new().prepend(3).prepend(2).prepend(1);
     // list is now of type &LinkedList, not ~LinkedList
     ```
    */
    pub fn prepend<'a>(&'a mut self, x: T) -> &'a mut LinkedList<T>{
        let old = ~util::replace(self, End);
        *self = Cell(x, old);
        self
    }

    /**
     Inserts the element x at position pos in the list.

     #Example
     ```rust
     use linked_list::LinkedList;
     let mut list = LinkedList::from_vec([1,2,4,5]);
     list.insert(3,2);
     assert!(list == LinkedList::from_vec([1,2,3,4,5]));
     ```

     #Failure
     The method fails if you try to insert an element after the end of the list
    */
    pub fn insert(&mut self, x: T, pos: uint){
        match pos{
            0  => {self.prepend(x);},
            _  => {
                match self {
                    &Cell(_, ref mut xs) => xs.insert(x, pos-1),
                    &End                 => fail!("You can't insert an object after the end of the list!")
                }
            }
        }
    }

    /**
     Replaces the object at pos by the object obj.

     #Example

     ```rust
     use linked_list::LinkedList;
     let mut list = LinkedList::from_vec([1,2,3,5]);
     list.replace(3, 4);
     assert!(list == LinkedList::from_vec([1,2,3,4]));   
     ```

     #Failure
     The method fails if pos is greater than the list's length
    */
    pub fn replace(&mut self, pos: uint, obj: T){
        match self{
            &Cell(ref mut x, ref mut xs) => if pos == 0 { *x = obj } else { xs.replace(pos-1, obj) },
            &End                         => fail!("You can't replace an element that doesn't exist")
        }
    }

    /**
     Appends a list at the end of the list this method is called on.
    
     #Example

     ```rust
     use linked_list::LinkedList;
     let mut list1 = LinkedList::from_vec([1,2]);
     let list2 = LinkedList::from_vec([3,4,5,6]);
     list1.append_list(list2);
     assert!(list1 == LinkedList::from_vec([1,2,3,4,5,6]));
     ```

     #Note
     This consumes the list passed as an argument
    */
    pub fn append_list(&mut self, list: ~LinkedList<T>){
        match self{
            &Cell(_, ref mut xs) if **xs == End => *xs = list,
            &Cell(_, ref mut xs)                => xs.append_list(list),
            &End                                => *self = *list
        }
    }

    /**
     Returns a copy of the element at position i of the linked list.

     list.at(i) is equivalent to list[i], this method is provided only for consistency with
     other at_() methods.

     #Failure
     This method fails if you try to access an element after the end of the list.
    */
    pub fn at(&self, i:uint) -> T{
        self[i]
    }

    /**
     Returns a reference to the element at position i of the linked list.

     #Failure
     This method fails if you try to access an element after the end of the list.
    */
    pub fn at_ref<'a>(&'a self, i:uint) -> &'a T {
        match self{
            &Cell(ref x, ref xs) => if i==0 {x} else {xs.at_ref(i-1)},
            &End                 => fail!("You can't index after the end of the list")
        }
    }

    /**
     Returns a mutable reference to the element at position i of the linked list.

     #Failure
     This method fails if you try to access an element after the end of the list.
    */
    pub fn at_mut_ref<'a>(&'a mut self, i:uint) -> &'a mut T{
        match self{
            &Cell(ref mut x, ref mut xs) => if i==0 {x} else {xs.at_mut_ref(i-1)},
            &End                         => fail!("You can't index after the end of the list")
        }
    }

    /**
     Produces an iterator over the LinkedList.
     Copies the elements of the list.

     #Example

     ```rust
     use linked_list::LinkedList;
     let list = LinkedList::from_vec([1,2,3,4,5]);
     assert!(list.iter().fold(0, |a,b| a+b) == 15);
     ```
    */
    pub fn iter<'a>(&'a self) -> LinkedListIterator<'a, T> {
        LinkedListIterator{current:self}
    }

    /**
     Produces a reference iterator over the LinkedList.
     Does not copy the elements of the list.

     #Example

     ```rust
     use linked_list::LinkedList;
     let list = LinkedList::from_vec([1,2,3,4,5]);
     assert!(list.ref_iter().fold(0, |a,&b| a+b) == 15);
     ```
    */
    pub fn ref_iter<'a>(&'a self) -> LinkedListRefIterator<'a, T> {
        LinkedListRefIterator{current:self}
    }

    /**
     Produces a moving iterator over the LinkedList.
     Does not copy the elements of the list.
     Consumes the list.

     #Example

     ```rust
     use linked_list::LinkedList;
     let mut list = LinkedList::from_vec([1,2,3,4,5]);
     assert!(list.move_iter().fold(0, |a,~b| a+b) == 15);
     ```

     You can therefore 

    */
    pub fn move_iter(mut ~self) -> LinkedListMoveIterator<T> {
        LinkedListMoveIterator{current:self}
    }
}

impl<'a, T: Clone> Iterator<T> for LinkedListIterator<'a, T>{ 
    fn next(&mut self) -> Option<T> {
        match self.current {
            &Cell(ref x, ref xs) => {self.current = &**xs; Some((*x).clone())},
            &End                 => None
        }
    } 
}

impl<'a, T: Clone> Iterator<&'a T> for LinkedListRefIterator<'a, T>{ 
    fn next(&mut self) -> Option<&'a T> {
        match self.current {
            &Cell(ref x, ref xs) => {self.current = &**xs; Some(x)},
            &End                 => None
        }
    } 
}

impl<T: Eq+Clone> Iterator<~T> for LinkedListMoveIterator<T>{ 
    fn next(&mut self) -> Option<~T> {
        let empty_iter = LinkedList::<T>::new().move_iter();
        let old = ~util::replace(self, empty_iter);

        match old.current {
            ~Cell(x, xs) => {self.current = xs; Some(~x)},
            ~End         => None
        }
    } 
}

#[cfg(test)]
mod test_linked_list{
    use super::{Cell, End, LinkedList};

    #[test]
    fn test_new(){
        let new_list = LinkedList::<int>::new();
        assert!(new_list.is_empty());
    }

    #[test]
    fn test_append(){
        let mut list = LinkedList::new();
        list.append(15).append(45);
        assert!(list == LinkedList::from_vec([15,45]));
    }

    #[test]
    fn test_len(){
        let empty_list = LinkedList::<int>::new();
        let list = LinkedList::from_vec([15,85]);
        assert!(empty_list.len() == 0);
        assert!(list.len() == 2);
    }

    #[test]
    fn test_from_vec(){
        let vec = [1,2,3];
        let list = LinkedList::from_vec(vec);
        assert!(list == ~Cell(1, ~Cell(2, ~Cell(3, ~End))));
    }

    #[test]
    fn test_prepend(){
        let mut list = LinkedList::new();
        list.prepend(5).prepend(4).prepend(3);
        assert!(list == LinkedList::from_vec([3,4,5]));
    }

    #[test]
    fn test_iter(){
        let vec = ~[1,2,3,4,5];
        let mut test_vec = ~[];
        let list = LinkedList::from_vec(vec);
        for i in list.iter(){
           test_vec.push(i);
        }

        assert!(vec == test_vec);
        assert!(list.iter().fold(0, |a, b| a+b) == 15);
    }

    #[test]
    fn test_ref_iter(){
        let vec = ~[1,2,3,4,5];
        let mut test_vec = ~[];
        let list = LinkedList::from_vec(vec);
        for &i in list.ref_iter(){
           test_vec.push(i);
        }

        assert!(vec == test_vec);
        assert!(list.ref_iter().fold(0, |a, &b| a+b) == 15);
    }

    #[test]
    fn test_move_iter(){
        let vec = ~[1,2,3,4,5];
        let mut test_vec = ~[];
        let mut list = LinkedList::from_vec(vec);

        for ~i in list.move_iter(){
           test_vec.push(i);
        }
        assert!(vec == test_vec);

        list = LinkedList::from_vec(vec);
        assert!(list.move_iter().fold(0, |a, ~b| a+b) == 15);
    }

    #[test]
    fn test_insert(){
        let mut list = LinkedList::from_vec([1,2,4,5]);
        list.insert(3,2);
        assert!(list == LinkedList::from_vec([1,2,3,4,5]));
    }

    #[test]
    fn test_indexing(){
        let list = LinkedList::from_vec([1,2,3,4]);
        assert!(list[2] == 3);
    }

    #[test]
    fn test_replace(){
        let mut list = LinkedList::from_vec([1,2,3,5]);
        list.replace(3, 4);
        assert!(list == LinkedList::from_vec([1,2,3,4]));
    }

    #[test]
    fn test_append_list(){
        let mut list1 = LinkedList::from_vec([1,2]);
        let list2 = LinkedList::from_vec([3,4,5,6]);
        list1.append_list(list2);
        assert!(list1 == LinkedList::from_vec([1,2,3,4,5,6]));
    }

    #[test]
    fn test_at(){
        let list = LinkedList::from_vec([1,2,3,4]);
        assert!(list.at(2) == 3);
    }

    #[test]
    fn test_at_ref(){
        let list = LinkedList::from_vec([1,2,3,4]);
        assert!(list.at_ref(2) == &3);
    }

    #[test]
    fn test_at_mut_ref(){
        let mut list = LinkedList::from_vec([1,2,3,4]);
        assert!(*list.at_mut_ref(2) == 3);
        *(list.at_mut_ref(2)) = 12;
        assert!(*list.at_mut_ref(2) == 12);
    }

    #[test]
    fn test_from_iterator(){
        let list = LinkedList::from_vec([1,2,3,4,5]);
        let list2: ~LinkedList<int> = list.iter().collect();
        assert!(list == list2);
    }
}

