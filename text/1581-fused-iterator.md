- Feature Name: fused
- Start Date: 2016-04-15
- RFC PR: [rust-lang/rfcs#1581](https://github.com/rust-lang/rfcs/pull/1581)
- Rust Issue: [rust-lang/rust#35602](https://github.com/rust-lang/rust/issues/35602)

# Summary
[summary]: #summary

Add a marker trait `FusedIterator` to `std::iter` and implement it on `Fuse<I>` and
applicable iterators and adapters. By implementing `FusedIterator`, an iterator
promises to behave as if `Iterator::fuse()` had been called on it (i.e. return
`None` forever after returning `None` once). Then, specialize `Fuse<I>` to be a
no-op if `I` implements `FusedIterator`.

# Motivation
[motivation]: #motivation

Iterators are allowed to return whatever they want after returning `None` once.
However, assuming that an iterator continues to return `None` can make
implementing some algorithms/adapters easier. Therefore, `Fuse` and
`Iterator::fuse` exist. Unfortunately, the `Fuse` iterator adapter introduces a
noticeable overhead. Furthermore, many iterators (most if not all iterators in
std) already act as if they were fused (this is considered to be the "polite"
behavior). Therefore, it would be nice to be able to pay the `Fuse` overhead
only when necessary.

Microbenchmarks:

```text
test fuse          ... bench:         200 ns/iter (+/- 13)
test fuse_fuse     ... bench:         250 ns/iter (+/- 10)
test myfuse        ... bench:          48 ns/iter (+/- 4)
test myfuse_myfuse ... bench:          48 ns/iter (+/- 3)
test range         ... bench:          48 ns/iter (+/- 2)
```

```rust
#![feature(test, specialization)]
extern crate test;

use std::ops::Range;

#[derive(Clone, Debug)]
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct Fuse<I> {
    iter: I,
    done: bool
}

pub trait FusedIterator: Iterator {}

trait IterExt: Iterator + Sized {
    fn myfuse(self) -> Fuse<Self> {
        Fuse {
            iter: self,
            done: false,
        }
    }
}

impl<I> FusedIterator for Fuse<I> where Fuse<I>: Iterator {}
impl<T> FusedIterator for Range<T> where Range<T>: Iterator {}

impl<T: Iterator> IterExt for T {}

impl<I> Iterator for Fuse<I> where I: Iterator {
    type Item = <I as Iterator>::Item;

    #[inline]
    default fn next(&mut self) -> Option<<I as Iterator>::Item> {
        if self.done {
            None
        } else {
            let next = self.iter.next();
            self.done = next.is_none();
            next
        }
    }
}

impl<I> Iterator for Fuse<I> where I: FusedIterator {
    #[inline]
    fn next(&mut self) -> Option<<I as Iterator>::Item> {
        self.iter.next()
    }
}

impl<I> ExactSizeIterator for Fuse<I> where I: ExactSizeIterator {}

#[bench]
fn myfuse(b: &mut test::Bencher) {
    b.iter(|| {
        for i in (0..100).myfuse() {
            test::black_box(i);
        }
    })
}

#[bench]
fn myfuse_myfuse(b: &mut test::Bencher) {
    b.iter(|| {
        for i in (0..100).myfuse().myfuse() {
            test::black_box(i);
        }
    });
}


#[bench]
fn fuse(b: &mut test::Bencher) {
    b.iter(|| {
        for i in (0..100).fuse() {
            test::black_box(i);
        }
    })
}

#[bench]
fn fuse_fuse(b: &mut test::Bencher) {
    b.iter(|| {
        for i in (0..100).fuse().fuse() {
            test::black_box(i);
        }
    });
}

#[bench]
fn range(b: &mut test::Bencher) {
    b.iter(|| {
        for i in (0..100) {
            test::black_box(i);
        }
    })
}
```

# Detailed Design
[design]: #detailed-design

```
trait FusedIterator: Iterator {}

impl<I: Iterator> FusedIterator for Fuse<I> {}

impl<A> FusedIterator for Range<A> {}
// ...and for most std/core iterators...


// Existing implementation of Fuse repeated for convenience
pub struct Fuse<I> {
    iterator: I,
    done: bool,
}

impl<I> Iterator for Fuse<I> where I: Iterator {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Self::Item {
        if self.done {
            None
        } else {
            let next = self.iterator.next();
            self.done = next.is_none();
            next
        }
    }
}

// Then, specialize Fuse...
impl<I> Iterator for Fuse<I> where I: FusedIterator {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Self::Item {
        // Ignore the done flag and pass through.
        // Note: this means that the done flag should *never* be exposed to the
        // user.
        self.iterator.next()
    }
}

```

# Drawbacks
[drawbacks]: #drawbacks

1. Yet another special iterator trait.
2. There is a useless done flag on no-op `Fuse` adapters.
3. Fuse isn't used very often anyways. However, I would argue that it should be
   used more often and people are just playing fast and loose. I'm hoping that
   making `Fuse` free when unneeded will encourage people to use it when they should.
4. This trait locks implementors into following the `FusedIterator` spec;
   removing the `FusedIterator` implementation would be a breaking change. This
   precludes future optimizations that take advantage of the fact that the
   behavior of an `Iterator` is undefined after it returns `None` the first
   time.


# Alternatives

## Do Nothing

Just pay the overhead on the rare occasions when fused is actually used.

## IntoFused

Use an associated type (and set it to `Self` for iterators that already provide
the fused guarantee) and an `IntoFused` trait:

```rust
#![feature(specialization)]
use std::iter::Fuse;

trait FusedIterator: Iterator {}

trait IntoFused: Iterator + Sized {
    type Fused: Iterator<Item = Self::Item>;
    fn into_fused(self) -> Self::Fused;
}

impl<T> IntoFused for T where T: Iterator {
    default type Fused = Fuse<Self>;
    default fn into_fused(self) -> Self::Fused {
        // Currently complains about a mismatched type but I think that's a
        // specialization bug.
        self.fuse()
    }
}

impl<T> IntoFused for T where T: FusedIterator {
    type Fused = Self;

    fn into_fused(self) -> Self::Fused {
        self
    }
}
```

For now, this doesn't actually compile because rust believes that the associated
type `Fused` could be specialized independent of the `into_fuse` function.

While this method gets rid of memory overhead of a no-op `Fuse` wrapper, it adds
complexity, needs to be implemented as a separate trait (because adding
associated types is a breaking change), and can't be used to optimize the
iterators returned from `Iterator::fuse` (users would *have* to call
`IntoFused::into_fused`).

## Associated Type

If we add the ability to condition associated types on `Self: Sized`, I believe
we can add them without it being a breaking change (associated types only need
to be fully specified on DSTs). If so (after fixing the bug in specialization
noted above), we could do the following:

```rust
trait Iterator {
    type Item;
    type Fuse: Iterator<Item=Self::Item> where Self: Sized = Fuse<Self>;
    fn fuse(self) -> Self::Fuse where Self: Sized {
        Fuse {
            done: false,
            iter: self,
        }
    }
    // ...
}
```

However, changing an iterator to take advantage of this would be a breaking
change.

# Unresolved questions
[unresolved]: #unresolved-questions

Should this trait be unsafe? I can't think of any way generic unsafe code could
end up relying on the guarantees of `FusedIterator`.

~~Also, it's possible to implement the specialized `Fuse` struct without a useless
`done` bool. Unfortunately, it's *very* messy. IMO, this is not worth it for now
and can always be fixed in the future as it doesn't change the `FusedIterator`
trait.~~ Resolved: It's not possible to remove the `done` bool without making
`Fuse` invariant.

