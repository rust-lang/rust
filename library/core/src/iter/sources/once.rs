use crate::iter::{FusedIterator, TrustedLen};

/// Creates an iterator that yields an element exactly once.
///
/// This is commonly used to adapt a single value into a [`chain()`] of other
/// kinds of iteration. Maybe you have an iterator that covers almost
/// everything, but you need an extra special case. Maybe you have a function
/// which works on iterators, but you only need to process one value.
///
/// [`chain()`]: Iterator::chain
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::iter;
///
/// // one is the loneliest number
/// let mut one = iter::once(1);
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
/// let dirs = fs::read_dir(".foo")?;
///
/// // we need to convert from an iterator of DirEntry-s to an iterator of
/// // PathBufs, so we use map
/// let dirs = dirs.map(|file| file.unwrap().path());
///
/// // now, our iterator just for our config file
/// let config = iter::once(PathBuf::from(".foorc"));
///
/// // chain the two iterators together into one big iterator
/// let files = dirs.chain(config);
///
/// // this will give us all of the files in .foo as well as .foorc
/// for f in files {
///     println!("{f:?}");
/// }
/// # std::io::Result::Ok(())
/// ```
#[stable(feature = "iter_once", since = "1.2.0")]
pub fn once<T>(value: T) -> Once<T> {
    Once { inner: Some(value).into_iter() }
}

/// An iterator that yields an element exactly once.
///
/// This `struct` is created by the [`once()`] function. See its documentation for more.
#[derive(Clone, Debug)]
#[stable(feature = "iter_once", since = "1.2.0")]
#[rustc_diagnostic_item = "IterOnce"]
pub struct Once<T> {
    inner: crate::option::IntoIter<T>,
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> Iterator for Once<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> DoubleEndedIterator for Once<T> {
    fn next_back(&mut self) -> Option<T> {
        self.inner.next_back()
    }
}

#[stable(feature = "iter_once", since = "1.2.0")]
impl<T> ExactSizeIterator for Once<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl<T> TrustedLen for Once<T> {}

#[stable(feature = "fused", since = "1.26.0")]
impl<T> FusedIterator for Once<T> {}
