//@ check-pass
#![warn(clippy::significant_drop_tightening)]

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

trait Scopable: Sized {
    type SubType: Scopable;
}

struct Subtree<T: Scopable>(ManuallyDrop<Box<Tree<T::SubType>>>);

impl<T: Scopable> Drop for Subtree<T> {
    fn drop(&mut self) {
        // SAFETY: The field cannot be used after we drop
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl<T: Scopable> Deref for Subtree<T> {
    type Target = Tree<T::SubType>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Scopable> DerefMut for Subtree<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

enum Tree<T: Scopable> {
    Group(Vec<Tree<T>>),
    Subtree(Subtree<T>),
    Leaf(T),
}

impl<T: Scopable> Tree<T> {
    fn foo(self) -> Self {
        self
    }
}

fn main() {}
