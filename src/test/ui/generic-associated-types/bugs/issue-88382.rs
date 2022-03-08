// check-fail
// known-bug

// This should pass, but has a missed normalization due to HRTB.

#![feature(generic_associated_types)]

trait Iterable {
    type Iterator<'a> where Self: 'a;
    fn iter(&self) -> Self::Iterator<'_>;
}

struct SomeImplementation();

impl Iterable for SomeImplementation {
    type Iterator<'a> = std::iter::Empty<usize>;
    fn iter(&self) -> Self::Iterator<'_> {
        std::iter::empty()
    }
}

fn do_something<I: Iterable>(i: I, mut f: impl for<'a> Fn(&mut I::Iterator<'a>)) {
    f(&mut i.iter());
}

fn main() {
    do_something(SomeImplementation(), |_| ());
    do_something(SomeImplementation(), test);
}

fn test<'a, I: Iterable>(_: &mut I::Iterator<'a>) {}
