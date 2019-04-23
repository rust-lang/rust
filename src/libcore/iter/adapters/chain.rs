use crate::ops::Try;
use crate::usize;

use super::super::{Iterator, DoubleEndedIterator, FusedIterator, TrustedLen};

/// An iterator that strings two iterators together.
///
/// This `struct` is created by the [`chain`] method on [`Iterator`]. See its
/// documentation for more.
///
/// [`chain`]: trait.Iterator.html#method.chain
/// [`Iterator`]: trait.Iterator.html
#[derive(Clone, Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chain<A, B> {
    a: A,
    b: B,
    state: ChainState,
}
impl<A, B> Chain<A, B> {
    pub(in super::super) fn new(a: A, b: B) -> Chain<A, B> {
        Chain { a, b, state: ChainState::Both }
    }
}

// The iterator protocol specifies that iteration ends with the return value
// `None` from `.next()` (or `.next_back()`) and it is unspecified what
// further calls return. The chain adaptor must account for this since it uses
// two subiterators.
//
//  It uses three states:
//
//  - Both: `a` and `b` are remaining
//  - Front: `a` remaining
//  - Back: `b` remaining
//
//  The fourth state (neither iterator is remaining) only occurs after Chain has
//  returned None once, so we don't need to store this state.
#[derive(Clone, Debug)]
enum ChainState {
    // both front and back iterator are remaining
    Both,
    // only front is remaining
    Front,
    // only back is remaining
    Back,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> Iterator for Chain<A, B> where
    A: Iterator,
    B: Iterator<Item = A::Item>
{
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => match self.a.next() {
                elt @ Some(..) => elt,
                None => {
                    self.state = ChainState::Back;
                    self.b.next()
                }
            },
            ChainState::Front => self.a.next(),
            ChainState::Back => self.b.next(),
        }
    }

    #[inline]
    #[rustc_inherit_overflow_checks]
    fn count(self) -> usize {
        match self.state {
            ChainState::Both => self.a.count() + self.b.count(),
            ChainState::Front => self.a.count(),
            ChainState::Back => self.b.count(),
        }
    }

    fn try_fold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R where
        Self: Sized, F: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        let mut accum = init;
        match self.state {
            ChainState::Both | ChainState::Front => {
                accum = self.a.try_fold(accum, &mut f)?;
                if let ChainState::Both = self.state {
                    self.state = ChainState::Back;
                }
            }
            _ => { }
        }
        if let ChainState::Back = self.state {
            accum = self.b.try_fold(accum, &mut f)?;
        }
        Try::from_ok(accum)
    }

    fn fold<Acc, F>(self, init: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        match self.state {
            ChainState::Both | ChainState::Front => {
                accum = self.a.fold(accum, &mut f);
            }
            _ => { }
        }
        match self.state {
            ChainState::Both | ChainState::Back => {
                accum = self.b.fold(accum, &mut f);
            }
            _ => { }
        }
        accum
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<A::Item> {
        match self.state {
            ChainState::Both | ChainState::Front => {
                for x in self.a.by_ref() {
                    if n == 0 {
                        return Some(x)
                    }
                    n -= 1;
                }
                if let ChainState::Both = self.state {
                    self.state = ChainState::Back;
                }
            }
            ChainState::Back => {}
        }
        if let ChainState::Back = self.state {
            self.b.nth(n)
        } else {
            None
        }
    }

    #[inline]
    fn find<P>(&mut self, mut predicate: P) -> Option<Self::Item> where
        P: FnMut(&Self::Item) -> bool,
    {
        match self.state {
            ChainState::Both => match self.a.find(&mut predicate) {
                None => {
                    self.state = ChainState::Back;
                    self.b.find(predicate)
                }
                v => v
            },
            ChainState::Front => self.a.find(predicate),
            ChainState::Back => self.b.find(predicate),
        }
    }

    #[inline]
    fn last(self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => {
                // Must exhaust a before b.
                let a_last = self.a.last();
                let b_last = self.b.last();
                b_last.or(a_last)
            },
            ChainState::Front => self.a.last(),
            ChainState::Back => self.b.last()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = a_lower.saturating_add(b_lower);

        let upper = match (a_upper, b_upper) {
            (Some(x), Some(y)) => x.checked_add(y),
            _ => None
        };

        (lower, upper)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<A, B> DoubleEndedIterator for Chain<A, B> where
    A: DoubleEndedIterator,
    B: DoubleEndedIterator<Item=A::Item>,
{
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        match self.state {
            ChainState::Both => match self.b.next_back() {
                elt @ Some(..) => elt,
                None => {
                    self.state = ChainState::Front;
                    self.a.next_back()
                }
            },
            ChainState::Front => self.a.next_back(),
            ChainState::Back => self.b.next_back(),
        }
    }

    fn try_rfold<Acc, F, R>(&mut self, init: Acc, mut f: F) -> R where
        Self: Sized, F: FnMut(Acc, Self::Item) -> R, R: Try<Ok=Acc>
    {
        let mut accum = init;
        match self.state {
            ChainState::Both | ChainState::Back => {
                accum = self.b.try_rfold(accum, &mut f)?;
                if let ChainState::Both = self.state {
                    self.state = ChainState::Front;
                }
            }
            _ => { }
        }
        if let ChainState::Front = self.state {
            accum = self.a.try_rfold(accum, &mut f)?;
        }
        Try::from_ok(accum)
    }

    fn rfold<Acc, F>(self, init: Acc, mut f: F) -> Acc
        where F: FnMut(Acc, Self::Item) -> Acc,
    {
        let mut accum = init;
        match self.state {
            ChainState::Both | ChainState::Back => {
                accum = self.b.rfold(accum, &mut f);
            }
            _ => { }
        }
        match self.state {
            ChainState::Both | ChainState::Front => {
                accum = self.a.rfold(accum, &mut f);
            }
            _ => { }
        }
        accum
    }

}

// Note: *both* must be fused to handle double-ended iterators.
#[stable(feature = "fused", since = "1.26.0")]
impl<A, B> FusedIterator for Chain<A, B>
    where A: FusedIterator,
          B: FusedIterator<Item=A::Item>,
{}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<A, B> TrustedLen for Chain<A, B>
    where A: TrustedLen, B: TrustedLen<Item=A::Item>,
{}
