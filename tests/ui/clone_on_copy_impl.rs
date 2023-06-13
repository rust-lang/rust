#![allow(clippy::incorrect_clone_impl_on_copy_type)]

use std::fmt;
use std::marker::PhantomData;

pub struct Key<T> {
    #[doc(hidden)]
    pub __name: &'static str,
    #[doc(hidden)]
    pub __phantom: PhantomData<T>,
}

impl<T> Copy for Key<T> {}

impl<T> Clone for Key<T> {
    fn clone(&self) -> Self {
        Key {
            __name: self.__name,
            __phantom: self.__phantom,
        }
    }
}

fn main() {}
