use crate::iter::{InPlaceIterable, Iterator};
use crate::ops::{ControlFlow, Try};

mod chain;
mod cloned;
mod copied;
mod cycle;
mod enumerate;
mod filter;
mod filter_map;
mod flatten;
mod fuse;
mod inspect;
mod intersperse;
mod map;
mod map_while;
mod peekable;
mod rev;
mod scan;
mod skip;
mod skip_while;
mod step_by;
mod take;
mod take_while;
mod zip;

pub use self::{
    chain::Chain, cycle::Cycle, enumerate::Enumerate, filter::Filter, filter_map::FilterMap,
    flatten::FlatMap, fuse::Fuse, inspect::Inspect, map::Map, peekable::Peekable, rev::Rev,
    scan::Scan, skip::Skip, skip_while::SkipWhile, take::Take, take_while::TakeWhile, zip::Zip,
};

#[stable(feature = "iter_cloned", since = "1.1.0")]
pub use self::cloned::Cloned;

#[stable(feature = "iterator_step_by", since = "1.28.0")]
pub use self::step_by::StepBy;

#[stable(feature = "iterator_flatten", since = "1.29.0")]
pub use self::flatten::Flatten;

#[stable(feature = "iter_copied", since = "1.36.0")]
pub use self::copied::Copied;

#[stable(feature = "iter_intersperse", since = "1.56.0")]
pub use self::intersperse::{Intersperse, IntersperseWith};

#[stable(feature = "iter_map_while", since = "1.57.0")]
pub use self::map_while::MapWhile;

#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::zip::TrustedRandomAccess;

#[unstable(feature = "trusted_random_access", issue = "none")]
pub use self::zip::TrustedRandomAccessNoCoerce;

#[unstable(feature = "iter_zip", issue = "83574")]
pub use self::zip::zip;

/// This trait provides transitive access to source-stage in an iterator-adapter pipeline
/// under the conditions that
/// * the iterator source `S` itself implements `SourceIter<Source = S>`
/// * there is a delegating implementation of this trait for each adapter in the pipeline between
///   the source and the pipeline consumer.
///
/// When the source is an owning iterator struct (commonly called `IntoIter`) then
/// this can be useful for specializing [`FromIterator`] implementations or recovering the
/// remaining elements after an iterator has been partially exhausted.
///
/// Note that implementations do not necessarily have to provide access to the inner-most
/// source of a pipeline. A stateful intermediate adapter might eagerly evaluate a part
/// of the pipeline and expose its internal storage as source.
///
/// The trait is unsafe because implementers must uphold additional safety properties.
/// See [`as_inner`] for details.
///
/// # Examples
///
/// Retrieving a partially consumed source:
///
/// ```
/// # #![feature(inplace_iteration)]
/// # use std::iter::SourceIter;
///
/// let mut iter = vec![9, 9, 9].into_iter().map(|i| i * i);
/// let _ = iter.next();
/// let mut remainder = std::mem::replace(unsafe { iter.as_inner() }, Vec::new().into_iter());
/// println!("n = {} elements remaining", remainder.len());
/// ```
///
/// [`FromIterator`]: crate::iter::FromIterator
/// [`as_inner`]: SourceIter::as_inner
#[unstable(issue = "none", feature = "inplace_iteration")]
#[doc(hidden)]
pub unsafe trait SourceIter {
    /// A source stage in an iterator pipeline.
    type Source: Iterator;

    /// Retrieve the source of an iterator pipeline.
    ///
    /// # Safety
    ///
    /// Implementations of must return the same mutable reference for their lifetime, unless
    /// replaced by a caller.
    /// Callers may only replace the reference when they stopped iteration and drop the
    /// iterator pipeline after extracting the source.
    ///
    /// This means iterator adapters can rely on the source not changing during
    /// iteration but they cannot rely on it in their Drop implementations.
    ///
    /// Implementing this method means adapters relinquish private-only access to their
    /// source and can only rely on guarantees made based on method receiver types.
    /// The lack of restricted access also requires that adapters must uphold the source's
    /// public API even when they have access to its internals.
    ///
    /// Callers in turn must expect the source to be in any state that is consistent with
    /// its public API since adapters sitting between it and the source have the same
    /// access. In particular an adapter may have consumed more elements than strictly necessary.
    ///
    /// The overall goal of these requirements is to let the consumer of a pipeline use
    /// * whatever remains in the source after iteration has stopped
    /// * the memory that has become unused by advancing a consuming iterator
    ///
    /// [`next()`]: Iterator::next()
    unsafe fn as_inner(&mut self) -> &mut Self::Source;
}

/// An iterator adapter that produces output as long as the underlying
/// iterator produces `Result::Ok` values.
///
/// If an error is encountered, the iterator stops and the error is
/// stored.
pub(crate) struct ResultShunt<'a, I, E> {
    iter: I,
    error: &'a mut Result<(), E>,
}

/// Process the given iterator as if it yielded a `T` instead of a
/// `Result<T, _>`. Any errors will stop the inner iterator and
/// the overall result will be an error.
pub(crate) fn process_results<I, T, E, F, U>(iter: I, mut f: F) -> Result<U, E>
where
    I: Iterator<Item = Result<T, E>>,
    for<'a> F: FnMut(ResultShunt<'a, I, E>) -> U,
{
    let mut error = Ok(());
    let shunt = ResultShunt { iter, error: &mut error };
    let value = f(shunt);
    error.map(|()| value)
}

impl<I, T, E> Iterator for ResultShunt<'_, I, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.find(|_| true)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.error.is_err() {
            (0, Some(0))
        } else {
            let (_, upper) = self.iter.size_hint();
            (0, upper)
        }
    }

    fn try_fold<B, F, R>(&mut self, init: B, mut f: F) -> R
    where
        F: FnMut(B, Self::Item) -> R,
        R: Try<Output = B>,
    {
        let error = &mut *self.error;
        self.iter
            .try_fold(init, |acc, x| match x {
                Ok(x) => ControlFlow::from_try(f(acc, x)),
                Err(e) => {
                    *error = Err(e);
                    ControlFlow::Break(try { acc })
                }
            })
            .into_try()
    }

    fn fold<B, F>(mut self, init: B, fold: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        #[inline]
        fn ok<B, T>(mut f: impl FnMut(B, T) -> B) -> impl FnMut(B, T) -> Result<B, !> {
            move |acc, x| Ok(f(acc, x))
        }

        self.try_fold(init, ok(fold)).unwrap()
    }
}

#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<S: Iterator, I, E> SourceIter for ResultShunt<'_, I, E>
where
    I: SourceIter<Source = S>,
{
    type Source = S;

    #[inline]
    unsafe fn as_inner(&mut self) -> &mut S {
        // SAFETY: unsafe function forwarding to unsafe function with the same requirements
        unsafe { SourceIter::as_inner(&mut self.iter) }
    }
}

// SAFETY: ResultShunt::next calls I::find, which has to advance `iter` in order to
// return `Some(_)`. Since `iter` has type `I: InPlaceIterable` it's guaranteed that
// at least one item will be moved out from the underlying source.
#[unstable(issue = "none", feature = "inplace_iteration")]
unsafe impl<I, T, E> InPlaceIterable for ResultShunt<'_, I, E> where
    I: Iterator<Item = Result<T, E>> + InPlaceIterable
{
}
