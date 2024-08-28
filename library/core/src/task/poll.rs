#![stable(feature = "futures_api", since = "1.36.0")]

use crate::convert;
use crate::ops::{self, ControlFlow};

/// Indicates whether a value is available or if the current task has been
/// scheduled to receive a wakeup instead.
///
/// This is returned by [`Future::poll`](core::future::Future::poll).
#[must_use = "this `Poll` may be a `Pending` variant, which should be handled"]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[lang = "Poll"]
#[stable(feature = "futures_api", since = "1.36.0")]
pub enum Poll<T> {
    /// Represents that a value is immediately ready.
    #[lang = "Ready"]
    #[stable(feature = "futures_api", since = "1.36.0")]
    Ready(#[stable(feature = "futures_api", since = "1.36.0")] T),

    /// Represents that a value is not ready yet.
    ///
    /// When a function returns `Pending`, the function *must* also
    /// ensure that the current task is scheduled to be awoken when
    /// progress can be made.
    #[lang = "Pending"]
    #[stable(feature = "futures_api", since = "1.36.0")]
    Pending,
}

impl<T> Poll<T> {
    /// Maps a `Poll<T>` to `Poll<U>` by applying a function to a contained value.
    ///
    /// # Examples
    ///
    /// Converts a <code>Poll<[String]></code> into a <code>Poll<[usize]></code>, consuming
    /// the original:
    ///
    /// [String]: ../../std/string/struct.String.html "String"
    /// ```
    /// # use core::task::Poll;
    /// let poll_some_string = Poll::Ready(String::from("Hello, World!"));
    /// // `Poll::map` takes self *by value*, consuming `poll_some_string`
    /// let poll_some_len = poll_some_string.map(|s| s.len());
    ///
    /// assert_eq!(poll_some_len, Poll::Ready(13));
    /// ```
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[inline]
    pub fn map<U, F>(self, f: F) -> Poll<U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Poll::Ready(t) => Poll::Ready(f(t)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Returns `true` if the poll is a [`Poll::Ready`] value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let x: Poll<u32> = Poll::Ready(2);
    /// assert_eq!(x.is_ready(), true);
    ///
    /// let x: Poll<u32> = Poll::Pending;
    /// assert_eq!(x.is_ready(), false);
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_poll", since = "1.49.0")]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub const fn is_ready(&self) -> bool {
        matches!(*self, Poll::Ready(_))
    }

    /// Returns `true` if the poll is a [`Pending`] value.
    ///
    /// [`Pending`]: Poll::Pending
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let x: Poll<u32> = Poll::Ready(2);
    /// assert_eq!(x.is_pending(), false);
    ///
    /// let x: Poll<u32> = Poll::Pending;
    /// assert_eq!(x.is_pending(), true);
    /// ```
    #[inline]
    #[rustc_const_stable(feature = "const_poll", since = "1.49.0")]
    #[stable(feature = "futures_api", since = "1.36.0")]
    pub const fn is_pending(&self) -> bool {
        !self.is_ready()
    }
}

impl<T, E> Poll<Result<T, E>> {
    /// Maps a `Poll<Result<T, E>>` to `Poll<Result<U, E>>` by applying a
    /// function to a contained `Poll::Ready(Ok)` value, leaving all other
    /// variants untouched.
    ///
    /// This function can be used to compose the results of two functions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let res: Poll<Result<u8, _>> = Poll::Ready("12".parse());
    /// let squared = res.map_ok(|n| n * n);
    /// assert_eq!(squared, Poll::Ready(Ok(144)));
    /// ```
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[inline]
    pub fn map_ok<U, F>(self, f: F) -> Poll<Result<U, E>>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(f(t))),
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Maps a `Poll::Ready<Result<T, E>>` to `Poll::Ready<Result<T, F>>` by
    /// applying a function to a contained `Poll::Ready(Err)` value, leaving all other
    /// variants untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let res: Poll<Result<u8, _>> = Poll::Ready("oops".parse());
    /// let res = res.map_err(|_| 0_u8);
    /// assert_eq!(res, Poll::Ready(Err(0)));
    /// ```
    #[stable(feature = "futures_api", since = "1.36.0")]
    #[inline]
    pub fn map_err<U, F>(self, f: F) -> Poll<Result<T, U>>
    where
        F: FnOnce(E) -> U,
    {
        match self {
            Poll::Ready(Ok(t)) => Poll::Ready(Ok(t)),
            Poll::Ready(Err(e)) => Poll::Ready(Err(f(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<T, E> Poll<Option<Result<T, E>>> {
    /// Maps a `Poll<Option<Result<T, E>>>` to `Poll<Option<Result<U, E>>>` by
    /// applying a function to a contained `Poll::Ready(Some(Ok))` value,
    /// leaving all other variants untouched.
    ///
    /// This function can be used to compose the results of two functions.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let res: Poll<Option<Result<u8, _>>> = Poll::Ready(Some("12".parse()));
    /// let squared = res.map_ok(|n| n * n);
    /// assert_eq!(squared, Poll::Ready(Some(Ok(144))));
    /// ```
    #[stable(feature = "poll_map", since = "1.51.0")]
    #[inline]
    pub fn map_ok<U, F>(self, f: F) -> Poll<Option<Result<U, E>>>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Poll::Ready(Some(Ok(t))) => Poll::Ready(Some(Ok(f(t)))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Maps a `Poll::Ready<Option<Result<T, E>>>` to
    /// `Poll::Ready<Option<Result<T, F>>>` by applying a function to a
    /// contained `Poll::Ready(Some(Err))` value, leaving all other variants
    /// untouched.
    ///
    /// This function can be used to pass through a successful result while handling
    /// an error.
    ///
    /// # Examples
    ///
    /// ```
    /// # use core::task::Poll;
    /// let res: Poll<Option<Result<u8, _>>> = Poll::Ready(Some("oops".parse()));
    /// let res = res.map_err(|_| 0_u8);
    /// assert_eq!(res, Poll::Ready(Some(Err(0))));
    /// ```
    #[stable(feature = "poll_map", since = "1.51.0")]
    #[inline]
    pub fn map_err<U, F>(self, f: F) -> Poll<Option<Result<T, U>>>
    where
        F: FnOnce(E) -> U,
    {
        match self {
            Poll::Ready(Some(Ok(t))) => Poll::Ready(Some(Ok(t))),
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(f(e)))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[stable(feature = "futures_api", since = "1.36.0")]
impl<T> From<T> for Poll<T> {
    /// Moves the value into a [`Poll::Ready`] to make a `Poll<T>`.
    ///
    /// # Example
    ///
    /// ```
    /// # use core::task::Poll;
    /// assert_eq!(Poll::from(true), Poll::Ready(true));
    /// ```
    fn from(t: T) -> Poll<T> {
        Poll::Ready(t)
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E> ops::Try for Poll<Result<T, E>> {
    type Output = Poll<T>;
    type Residual = Result<convert::Infallible, E>;

    #[inline]
    fn from_output(c: Self::Output) -> Self {
        c.map(Ok)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Poll::Ready(Ok(x)) => ControlFlow::Continue(Poll::Ready(x)),
            Poll::Ready(Err(e)) => ControlFlow::Break(Err(e)),
            Poll::Pending => ControlFlow::Continue(Poll::Pending),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E, F: From<E>> ops::FromResidual<Result<convert::Infallible, E>> for Poll<Result<T, F>> {
    #[inline]
    fn from_residual(x: Result<convert::Infallible, E>) -> Self {
        match x {
            Err(e) => Poll::Ready(Err(From::from(e))),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E> ops::Try for Poll<Option<Result<T, E>>> {
    type Output = Poll<Option<T>>;
    type Residual = Result<convert::Infallible, E>;

    #[inline]
    fn from_output(c: Self::Output) -> Self {
        c.map(|x| x.map(Ok))
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Poll::Ready(Some(Ok(x))) => ControlFlow::Continue(Poll::Ready(Some(x))),
            Poll::Ready(Some(Err(e))) => ControlFlow::Break(Err(e)),
            Poll::Ready(None) => ControlFlow::Continue(Poll::Ready(None)),
            Poll::Pending => ControlFlow::Continue(Poll::Pending),
        }
    }
}

#[unstable(feature = "try_trait_v2", issue = "84277")]
impl<T, E, F: From<E>> ops::FromResidual<Result<convert::Infallible, E>>
    for Poll<Option<Result<T, F>>>
{
    #[inline]
    fn from_residual(x: Result<convert::Infallible, E>) -> Self {
        match x {
            Err(e) => Poll::Ready(Some(Err(From::from(e)))),
        }
    }
}
