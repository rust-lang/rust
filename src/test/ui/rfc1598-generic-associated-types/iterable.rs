#![feature(generic_associated_types)]
//~^ WARNING the feature `generic_associated_types` is incomplete

use std::ops::Deref;

// FIXME(#44265): "lifetime arguments are not allowed for this type" errors will be addressed in a
// follow-up PR.

trait Iterable {
    type Item<'a>;
    type Iter<'a>: Iterator<Item = Self::Item<'a>>;
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
}

// Impl for struct type
impl<T> Iterable for Vec<T> {
    type Item<'a> = &'a T;
    type Iter<'a> = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
        self.iter()
    }
}

// Impl for a primitive type
impl<T> Iterable for [T] {
    type Item<'a> = &'a T;
    type Iter<'a> = std::slice::Iter<'a, T>;

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
        self.iter()
    }
}

fn make_iter<'a, I: Iterable>(it: &'a I) -> I::Iter<'a> {
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
    it.iter()
}

fn get_first<'a, I: Iterable>(it: &'a I) -> Option<I::Item<'a>> {
    //~^ ERROR lifetime arguments are not allowed for this type [E0109]
    it.iter().next()
}

fn main() {}
