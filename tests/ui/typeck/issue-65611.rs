use std::mem::MaybeUninit;
use std::ops::Deref;

pub unsafe trait Array {
    /// The array’s element type
    type Item;
    #[doc(hidden)]
    /// The smallest index type that indexes the array.
    type Index: Index;
    #[doc(hidden)]
    fn as_ptr(&self) -> *const Self::Item;
    #[doc(hidden)]
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
    #[doc(hidden)]
    fn capacity() -> usize;
}

pub trait Index : PartialEq + Copy {
    fn to_usize(self) -> usize;
    fn from(i: usize) -> Self;
}

impl Index for usize {
    fn to_usize(self) -> usize { self }
    fn from(val: usize) -> Self {
        val
    }
}

unsafe impl<T> Array for [T; 1] {
    type Item = T;
    type Index = usize;
    fn as_ptr(&self) -> *const T { self as *const _ as *const _ }
    fn as_mut_ptr(&mut self) -> *mut T { self as *mut _ as *mut _}
    fn capacity() -> usize { 1 }
}

impl<A: Array> Deref for ArrayVec<A> {
    type Target = [A::Item];
    #[inline]
    fn deref(&self) -> &[A::Item] {
        panic!()
    }
}

pub struct ArrayVec<A: Array> {
    xs: MaybeUninit<A>,
    len: usize,
}

impl<A: Array> ArrayVec<A> {
    pub fn new() -> ArrayVec<A> {
        panic!()
    }
}

fn main() {
    let mut buffer = ArrayVec::new();
    let x = buffer.last().unwrap().0.clone();
    //~^ ERROR type annotations needed
    buffer.reverse();
}
