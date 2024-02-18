//@ check-pass

#![feature(associated_type_bounds)]

pub struct Flatten<I>
where
    I: Iterator<Item: IntoIterator>,
{
    inner: <I::Item as IntoIterator>::IntoIter,
}

fn main() {}
