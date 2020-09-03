use crate::ops::Try;

/// Used to make try_fold closures more like normal loops
#[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlFlow<C, B> {
    /// Continue in the loop, using the given value for the next iteration
    Continue(C),
    /// Exit the loop, yielding the given value
    Break(B),
}

#[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
impl<C, B> Try for ControlFlow<C, B> {
    type Ok = C;
    type Error = B;
    #[inline]
    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        match self {
            ControlFlow::Continue(y) => Ok(y),
            ControlFlow::Break(x) => Err(x),
        }
    }
    #[inline]
    fn from_error(v: Self::Error) -> Self {
        ControlFlow::Break(v)
    }
    #[inline]
    fn from_ok(v: Self::Ok) -> Self {
        ControlFlow::Continue(v)
    }
}

impl<C, B> ControlFlow<C, B> {
    /// Converts the `ControlFlow` into an `Option` which is `Some` if the
    /// `ControlFlow` was `Break` and `None` otherwise.
    #[inline]
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    pub fn break_value(self) -> Option<B> {
        match self {
            ControlFlow::Continue(..) => None,
            ControlFlow::Break(x) => Some(x),
        }
    }
}

impl<R: Try> ControlFlow<R::Ok, R> {
    /// Create a `ControlFlow` from any type implementing `Try`.
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    #[inline]
    pub fn from_try(r: R) -> Self {
        match Try::into_result(r) {
            Ok(v) => ControlFlow::Continue(v),
            Err(v) => ControlFlow::Break(Try::from_error(v)),
        }
    }

    /// Convert a `ControlFlow` into any type implementing `Try`;
    #[unstable(feature = "control_flow_enum", reason = "new API", issue = "75744")]
    #[inline]
    pub fn into_try(self) -> R {
        match self {
            ControlFlow::Continue(v) => Try::from_ok(v),
            ControlFlow::Break(v) => v,
        }
    }
}
