use crate::future::Future;

/// Conversion into a `Future`.
#[unstable(feature = "into_future", issue = "67644")]
pub trait IntoFuture {
    /// The output that the future will produce on completion.
    #[unstable(feature = "into_future", issue = "67644")]
    type Output;

    /// Which kind of future are we turning this into?
    #[unstable(feature = "into_future", issue = "67644")]
    type IntoFuture: Future<Output = Self::Output>;

    /// Creates a future from a value.
    #[unstable(feature = "into_future", issue = "67644")]
    #[lang = "into_future"]
    fn into_future(self) -> Self::IntoFuture;
}

#[unstable(feature = "into_future", issue = "67644")]
impl<F: Future> IntoFuture for F {
    type Output = F::Output;
    type IntoFuture = F;

    fn into_future(self) -> Self::IntoFuture {
        self
    }
}
