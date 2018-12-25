#![unstable(feature = "futures_api",
            reason = "futures in libcore are unstable",
            issue = "50547")]

use ops::Try;
use result::Result;

/// Indicates whether a value is available or if the current task has been
/// scheduled to receive a wakeup instead.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Poll<T> {
    /// Represents that a value is immediately ready.
    Ready(T),

    /// Represents that a value is not ready yet.
    ///
    /// When a function returns `Pending`, the function *must* also
    /// ensure that the current task is scheduled to be awoken when
    /// progress can be made.
    Pending,
}

impl<T> Poll<T> {
    /// Change the ready value of this `Poll` with the closure provided
    pub fn map<U, F>(self, f: F) -> Poll<U>
        where F: FnOnce(T) -> U
    {
        match self {
            Poll::Ready(t) => Poll::Ready(f(t)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Returns whether this is `Poll::Ready`
    #[inline]
    pub fn is_ready(&self) -> bool {
        match *self {
            Poll::Ready(_) => true,
            Poll::Pending => false,
        }
    }

    /// Returns whether this is `Poll::Pending`
    #[inline]
    pub fn is_pending(&self) -> bool {
        !self.is_ready()
    }
}

impl<T, E> Poll<Result<T, E>> {
    /// Change the success value of this `Poll` with the closure provided
    pub fn map_ok<U, F>(self, f: F) -> Poll<Result<U, E>>
        where F: FnOnce(T) -> U
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(f(t))),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Change the error value of this `Poll` with the closure provided
    pub fn map_err<U, F>(self, f: F) -> Poll<Result<T, U>>
        where F: FnOnce(E) -> U
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(t)),
            Poll::Ready(Err(e)) => Poll::Ready(Err(f(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T> From<T> for Poll<T> {
    fn from(t: T) -> Poll<T> {
        Poll::Ready(t)
    }
}

impl<T, E> Try for Poll<Result<T, E>> {
    type Ok = Poll<T>;
    type Error = E;

    #[inline]
    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        match self {
            Poll::Ready(Ok(x)) => Ok(Poll::Ready(x)),
            Poll::Ready(Err(e)) => Err(e),
            Poll::Pending => Ok(Poll::Pending),
        }
    }

    #[inline]
    fn from_error(e: Self::Error) -> Self {
        Poll::Ready(Err(e))
    }

    #[inline]
    fn from_ok(x: Self::Ok) -> Self {
        x.map(Ok)
    }
}

impl<T, E> Try for Poll<Option<Result<T, E>>> {
    type Ok = Poll<Option<T>>;
    type Error = E;

    #[inline]
    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        match self {
            Poll::Ready(Some(Ok(x))) => Ok(Poll::Ready(Some(x))),
            Poll::Ready(Some(Err(e))) => Err(e),
            Poll::Ready(None) => Ok(Poll::Ready(None)),
            Poll::Pending => Ok(Poll::Pending),
        }
    }

    #[inline]
    fn from_error(e: Self::Error) -> Self {
        Poll::Ready(Some(Err(e)))
    }

    #[inline]
    fn from_ok(x: Self::Ok) -> Self {
        x.map(|x| x.map(Ok))
    }
}
