// run-pass
#![allow(dead_code)]

use std::marker::PhantomData;

pub trait Consumer<Item> {
    type Result;
}

pub trait IndexedParallelIterator: ExactSizeIterator {
    type Item;
}

pub struct CollectConsumer<'c, T: Send> {
    target: &'c mut [T],
}

impl<'c, T: Send + 'c> Consumer<T> for CollectConsumer<'c, T> {
    type Result = CollectResult<'c, T>;
}

pub struct CollectResult<'c, T> {
    start: *mut T,
    len: usize,
    invariant_lifetime: PhantomData<&'c mut &'c mut [T]>,
}

unsafe impl<'c, T> Send for CollectResult<'c, T> where T: Send {}

pub fn unzip_indexed<I, A, B, CA>(_: I, _: CA) -> CA::Result
where
    I: IndexedParallelIterator<Item = (A, B)>,
    CA: Consumer<A>,
{
    unimplemented!()
}

struct Collect<'c, T: Send> {
    vec: &'c mut Vec<T>,
    len: usize,
}

pub fn unzip_into_vecs<I, A, B>(pi: I, left: &mut Vec<A>, _: &mut Vec<B>)
where
    I: IndexedParallelIterator<Item = (A, B)>,
    A: Send,
    B: Send,
{
    let len = pi.len();
    Collect::new(left, len).with_consumer(|left_consumer| unzip_indexed(pi, left_consumer));
}

impl<'c, T: Send + 'c> Collect<'c, T> {
    fn new(vec: &'c mut Vec<T>, len: usize) -> Self {
        Collect { vec, len }
    }

    fn with_consumer<F>(self, _: F)
    where
        F: FnOnce(CollectConsumer<T>) -> CollectResult<T>,
    {
        unimplemented!()
    }
}

fn main() {}
