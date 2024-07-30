use crate::future::Future;

/// Conversion into a `Future`.
///
/// By implementing `IntoFuture` for a type, you define how it will be
/// converted to a future.
///
/// # `.await` desugaring
///
/// The `.await` keyword desugars into a call to `IntoFuture::into_future`
/// first before polling the future to completion. `IntoFuture` is implemented
/// for all `T: Future` which means the `into_future` method will be available
/// on all futures.
///
/// ```no_run
/// use std::future::IntoFuture;
///
/// # async fn foo() {
/// let v = async { "meow" };
/// let mut fut = v.into_future();
/// assert_eq!("meow", fut.await);
/// # }
/// ```
///
/// # Async builders
///
/// When implementing futures manually there will often be a choice between
/// implementing `Future` or `IntoFuture` for a type. Implementing `Future` is a
/// good choice in most cases. But implementing `IntoFuture` is most useful when
/// implementing "async builder" types, which allow their values to be modified
/// multiple times before being `.await`ed.
///
/// ```rust
/// use std::future::{ready, Ready, IntoFuture};
///
/// /// Eventually multiply two numbers
/// pub struct Multiply {
///     num: u16,
///     factor: u16,
/// }
///
/// impl Multiply {
///     /// Constructs a new instance of `Multiply`.
///     pub fn new(num: u16, factor: u16) -> Self {
///         Self { num, factor }
///     }
///
///     /// Set the number to multiply by the factor.
///     pub fn number(mut self, num: u16) -> Self {
///         self.num = num;
///         self
///     }
///
///     /// Set the factor to multiply the number with.
///     pub fn factor(mut self, factor: u16) -> Self {
///         self.factor = factor;
///         self
///     }
/// }
///
/// impl IntoFuture for Multiply {
///     type Output = u16;
///     type IntoFuture = Ready<Self::Output>;
///
///     fn into_future(self) -> Self::IntoFuture {
///         ready(self.num * self.factor)
///     }
/// }
///
/// // NOTE: Rust does not yet have an `async fn main` function, that functionality
/// // currently only exists in the ecosystem.
/// async fn run() {
///     let num = Multiply::new(0, 0)  // initialize the builder to number: 0, factor: 0
///         .number(2)                 // change the number to 2
///         .factor(2)                 // change the factor to 2
///         .await;                    // convert to future and .await
///
///     assert_eq!(num, 4);
/// }
/// ```
///
/// # Usage in trait bounds
///
/// Using `IntoFuture` in trait bounds allows a function to be generic over both
/// `Future` and `IntoFuture`. This is convenient for users of the function, so
/// when they are using it they don't have to make an extra call to
/// `IntoFuture::into_future` to obtain an instance of `Future`:
///
/// ```rust
/// use std::future::IntoFuture;
///
/// /// Converts the output of a future to a string.
/// async fn fut_to_string<Fut>(fut: Fut) -> String
/// where
///     Fut: IntoFuture,
///     Fut::Output: std::fmt::Debug,
/// {
///     format!("{:?}", fut.await)
/// }
/// ```
#[stable(feature = "into_future", since = "1.64.0")]
#[rustc_diagnostic_item = "IntoFuture"]
#[diagnostic::on_unimplemented(
    label = "`{Self}` is not a future",
    message = "`{Self}` is not a future",
    note = "{Self} must be a future or must implement `IntoFuture` to be awaited"
)]
pub trait IntoFuture {
    /// The output that the future will produce on completion.
    #[stable(feature = "into_future", since = "1.64.0")]
    type Output;

    /// Which kind of future are we turning this into?
    #[stable(feature = "into_future", since = "1.64.0")]
    type IntoFuture: Future<Output = Self::Output>;

    /// Creates a future from a value.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```no_run
    /// use std::future::IntoFuture;
    ///
    /// # async fn foo() {
    /// let v = async { "meow" };
    /// let mut fut = v.into_future();
    /// assert_eq!("meow", fut.await);
    /// # }
    /// ```
    #[stable(feature = "into_future", since = "1.64.0")]
    #[lang = "into_future"]
    fn into_future(self) -> Self::IntoFuture;
}

#[stable(feature = "into_future", since = "1.64.0")]
impl<F: Future> IntoFuture for F {
    type Output = F::Output;
    type IntoFuture = F;

    fn into_future(self) -> Self::IntoFuture {
        self
    }
}
