use crate::fmt;
use crate::iter::{FusedIterator, TrustedLen};

/// Creates an iterator that lazily generates a value exactly once by invoking
/// the provided closure.
///
/// This is commonly used to adapt a single value generator into a [`chain()`] of
/// other kinds of iteration. Maybe you have an iterator that covers almost
/// everything, but you need an extra special case. Maybe you have a function
/// which works on iterators, but you only need to process one value.
///
/// Unlike [`once()`], this function will lazily generate the value on request.
///
/// [`chain()`]: Iterator::chain
/// [`once()`]: crate::iter::once
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // one is the loneliest number
/// let mut one = iter::once_with(|| 1);
///
/// assert_eq!(Some(1), one.next());
///
/// // just one, that's all we get
/// assert_eq!(None, one.next());
/// ```
///
/// Chaining together with another iterator. Let's say that we want to iterate
/// over each file of the `.foo` directory, but also a configuration file,
/// `.foorc`:
///
/// ```no_run
/// use std::iter;
/// use std::fs;
/// use std::path::PathBuf;
///
/// let dirs = fs::read_dir(".foo").unwrap();
///
/// // we need to convert from an iterator of DirEntry-s to an iterator of
/// // PathBufs, so we use map
/// let dirs = dirs.map(|file| file.unwrap().path());
///
/// // now, our iterator just for our config file
/// let config = iter::once_with(|| PathBuf::from(".foorc"));
///
/// // chain the two iterators together into one big iterator
/// let files = dirs.chain(config);
///
/// // this will give us all of the files in .foo as well as .foorc
/// for f in files {
///     println!("{f:?}");
/// }
/// ```
#[inline]
#[stable(feature = "iter_once_with", since = "1.43.0")]
pub fn once_with<A, F: FnOnce() -> A>(gen: F) -> OnceWith<F> {
    OnceWith { gen: Some(gen) }
}

/// An iterator that yields a single element of type `A` by
/// applying the provided closure `F: FnOnce() -> A`.
///
/// This `struct` is created by the [`once_with()`] function.
/// See its documentation for more.
#[derive(Clone)]
#[stable(feature = "iter_once_with", since = "1.43.0")]
pub struct OnceWith<F> {
    gen: Option<F>,
}

#[stable(feature = "iter_once_with_debug", since = "1.68.0")]
impl<F> fmt::Debug for OnceWith<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.gen.is_some() {
            f.write_str("OnceWith(Some(_))")
        } else {
            f.write_str("OnceWith(None)")
        }
    }
}

#[stable(feature = "iter_once_with", since = "1.43.0")]
impl<A, F: FnOnce() -> A> Iterator for OnceWith<F> {
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        let f = self.gen.take()?;
        Some(f())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.gen.iter().size_hint()
    }
}

#[stable(feature = "iter_once_with", since = "1.43.0")]
impl<A, F: FnOnce() -> A> DoubleEndedIterator for OnceWith<F> {
    fn next_back(&mut self) -> Option<A> {
        self.next()
    }
}

#[stable(feature = "iter_once_with", since = "1.43.0")]
impl<A, F: FnOnce() -> A> ExactSizeIterator for OnceWith<F> {
    fn len(&self) -> usize {
        self.gen.iter().len()
    }
}

#[stable(feature = "iter_once_with", since = "1.43.0")]
impl<A, F: FnOnce() -> A> FusedIterator for OnceWith<F> {}

#[stable(feature = "iter_once_with", since = "1.43.0")]
unsafe impl<A, F: FnOnce() -> A> TrustedLen for OnceWith<F> {}
