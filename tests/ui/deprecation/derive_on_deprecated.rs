//@ build-pass (FIXME(62277): could be check-pass?)

#![deny(deprecated)]

#[deprecated = "oh no"]
#[derive(Default)]
struct X;

#[deprecated(note="Do not use this")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct Step<I> {
    _skip: Option<I>,
}

fn main() {}
