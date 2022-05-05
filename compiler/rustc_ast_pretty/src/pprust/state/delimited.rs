use std::iter::Peekable;
use std::mem;
use std::ops::Deref;

pub struct Delimited<I: Iterator> {
    is_first: bool,
    iter: Peekable<I>,
}

pub trait IterDelimited: Iterator + Sized {
    fn delimited(self) -> Delimited<Self> {
        Delimited { is_first: true, iter: self.peekable() }
    }
}

impl<I: Iterator> IterDelimited for I {}

pub struct IteratorItem<T> {
    value: T,
    pub is_first: bool,
    pub is_last: bool,
}

impl<I: Iterator> Iterator for Delimited<I> {
    type Item = IteratorItem<I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.iter.next()?;
        let is_first = mem::replace(&mut self.is_first, false);
        let is_last = self.iter.peek().is_none();
        Some(IteratorItem { value, is_first, is_last })
    }
}

impl<T> Deref for IteratorItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
