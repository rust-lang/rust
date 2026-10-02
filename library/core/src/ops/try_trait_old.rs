//! Internal lang items for historical `?` desugaring

use crate::ops::ControlFlow;
use crate::option::Option;
use crate::result::Result;
use crate::task::Poll;

//FIXME: fix messages - revert changes tests/ui/try-trait (currently updated to pass with bad messages)
#[rustc_on_unimplemented(
    on(
        all(from_desugaring = "TryBlock"),
        message = "a `try` block must return `Result` or `Option` \
                    (or another type that implements `{This}`)",
        label = "could not wrap the final value of the block as `{Self}` doesn't implement `Try`",
    ),
    on(
        all(from_desugaring = "QuestionMark"),
        message = "the `?` operator can only be applied to values that implement `{This}`",
        label = "the `?` operator cannot be applied to type `{Self}`"
    )
)]
#[lang = "Try_Old"]
#[stable(feature = "stability_marker_for_builtins", since = "0.0.0")]
#[expect(unreachable_pub, reason = "lang_items")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait Try: [const] FromResidual {
    /// The type of the value produced by `?` when *not* short-circuiting.
    type Output;

    /// The type of the value passed to [`FromResidual::from_residual`]
    /// as part of `?` when short-circuiting.
    ///
    /// This represents the possible values of the `Self` type which are *not*
    /// represented by the `Output` type.
    type Residual: Residual<Self::Output>;

    /// Constructs the type from its `Output` type.
    ///
    /// This should be implemented consistently with the `branch` method
    /// such that applying the `?` operator will get back the original value:
    /// `Try::from_output(x).branch() --> ControlFlow::Continue(x)`.
    ///
    /// This is only used in `try` blocks. It does not need to be supported for
    /// historical `?`
    #[expect(dead_code, reason = "only required for try blocks, which do not support legacy `?`")]
    fn from_output(_output: Self::Output) -> Self;

    /// Used in `?` to decide whether the operator should produce a value
    /// (because this returned [`ControlFlow::Continue`])
    /// or propagate a value back to the caller
    /// (because this returned [`ControlFlow::Break`]).
    #[lang = "branch_old"]
    #[stable(feature = "stability_marker_for_builtins", since = "0.0.0")]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output>;
}

/// Used to specify which residuals can be converted into which [`crate::ops::Try`] types.
///
/// Every `Try` type needs to be recreatable from its own associated
/// `Residual` type, but can also have additional `FromResidual` implementations
/// to support interconversion with other `Try` types.
#[rustc_on_unimplemented(
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::result::Result<T, E>",
            R = "core::option::Option<!>",
        ),
        message = "the `?` operator can only be used on `Result`s, not `Option`s, \
            in {ItemContext} that returns `Result`",
        label = "use `.ok_or(...)?` to provide an error compatible with `{Self}`",
        parent_label = "this function returns a `Result`"
    ),
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::result::Result<T, E>",
        ),
        // There's a special error message in the trait selection code for
        // `From` in `?`, so this is not shown for result-in-result errors,
        // and thus it can be phrased more strongly than `ControlFlow`'s.
        message = "the `?` operator can only be used on `Result`s \
            in {ItemContext} that returns `Result`",
        label = "this `?` produces `{R}`, which is incompatible with `{Self}`",
        parent_label = "this function returns a `Result`"
    ),
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::option::Option<T>",
            R = "core::result::Result<T, E>",
        ),
        message = "the `?` operator can only be used on `Option`s, not `Result`s, \
            in {ItemContext} that returns `Option`",
        label = "use `.ok()?` if you want to discard the `{R}` error information",
        parent_label = "this function returns an `Option`"
    ),
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::option::Option<T>",
        ),
        // `Option`-in-`Option` always works, as there's only one possible
        // residual, so this can also be phrased strongly.
        message = "the `?` operator can only be used on `Option`s \
            in {ItemContext} that returns `Option`",
        label = "this `?` produces `{R}`, which is incompatible with `{Self}`",
        parent_label = "this function returns an `Option`"
    ),
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::ops::control_flow::ControlFlow<B, C>",
            R = "core::ops::control_flow::ControlFlow<B, C>",
        ),
        message = "the `?` operator in {ItemContext} that returns `ControlFlow<B, _>` \
            can only be used on other `ControlFlow<B, _>`s (with the same Break type)",
        label = "this `?` produces `{R}`, which is incompatible with `{Self}`",
        parent_label = "this function returns a `ControlFlow`",
        note = "unlike `Result`, there's no `From`-conversion performed for `ControlFlow`"
    ),
    on(
        all(
            from_desugaring = "QuestionMark",
            Self = "core::ops::control_flow::ControlFlow<B, C>",
            // `R` is not a `ControlFlow`, as that case was matched previously
        ),
        message = "the `?` operator can only be used on `ControlFlow`s \
            in {ItemContext} that returns `ControlFlow`",
        label = "this `?` produces `{R}`, which is incompatible with `{Self}`",
        parent_label = "this function returns a `ControlFlow`",
    ),
    on(
        all(from_desugaring = "QuestionMark"),
        message = "the `?` operator can only be used in {ItemContext} \
                    that returns `Result` or `Option` \
                    (or another type that implements `{This}`)",
        label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
        parent_label = "this function should return `Result` or `Option` to accept `?`"
    ),
)]
#[rustc_diagnostic_item = "FromResidualOld"]
#[expect(unreachable_pub, reason = "lang_items")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait FromResidual<R = <Self as Try>::Residual> {
    /// Constructs the type from a compatible `Residual` type.
    ///
    /// This should be implemented consistently with the `branch` method such
    /// that applying the `?` operator will get back an equivalent residual:
    /// `FromResidual::from_residual(r).branch() --> ControlFlow::Break(r)`.
    /// (The residual is not mandated to be *identical* when interconversion is involved.)
    #[lang = "from_residual_old"]
    #[stable(feature = "stability_marker_for_builtins", since = "0.0.0")]
    fn from_residual(residual: R) -> Self;
}

/// Allows retrieving the canonical type implementing [`Try`] that has this type
/// as its residual and allows it to hold an `O` as its output.
///
/// If you think of the `Try` trait as splitting a type into its [`Try::Output`]
/// and [`Try::Residual`] components, this allows putting them back together.
///
/// For example,
/// `Result<T, E>: Try<Output = T, Residual = Result<!, E>>`,
/// and in the other direction,
/// `<Result<!, E> as Residual<T>>::TryType = Result<T, E>`.
#[expect(unreachable_pub, reason = "lang_items")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait Residual<O>: Sized {
    /// The "return" type of this meta-function.
    type TryType: Try<Output = O, Residual = Self>;
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T, E> Try for Result<T, E> {
    type Output = T;
    type Residual = Result<!, E>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Ok(v) => ControlFlow::Continue(v),
            Err(e) => ControlFlow::Break(Err(e)),
        }
    }
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T, E, F: [const] From<E>> FromResidual<Result<!, E>> for Result<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: Result<!, E>) -> Self {
        match residual {
            Err(e) => Err(From::from(e)),
        }
    }
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T, E> Residual<T> for Result<!, E> {
    type TryType = Result<T, E>;
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<B, C> Try for ControlFlow<B, C> {
    type Output = C;
    type Residual = ControlFlow<B, !>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        ControlFlow::Continue(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            ControlFlow::Continue(c) => ControlFlow::Continue(c),
            ControlFlow::Break(b) => ControlFlow::Break(ControlFlow::Break(b)),
        }
    }
}

// Note: manually specifying the residual type instead of using the default to work around
// https://github.com/rust-lang/rust/issues/99940
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<B, C> FromResidual<ControlFlow<B, !>> for ControlFlow<B, C> {
    #[inline]
    fn from_residual(residual: ControlFlow<B, !>) -> Self {
        match residual {
            ControlFlow::Break(b) => ControlFlow::Break(b),
        }
    }
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<B, C> Residual<C> for ControlFlow<B, !> {
    type TryType = ControlFlow<B, C>;
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T> Try for Option<T> {
    type Output = T;
    type Residual = Option<!>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Some(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Some(v) => ControlFlow::Continue(v),
            None => ControlFlow::Break(None),
        }
    }
}

// Note: manually specifying the residual type instead of using the default to work around
// https://github.com/rust-lang/rust/issues/99940
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T> FromResidual<Option<!>> for Option<T> {
    #[inline]
    fn from_residual(residual: Option<!>) -> Self {
        match residual {
            None => None,
        }
    }
}

#[rustc_const_unstable(feature = "const_try", issue = "74935")]
const impl<T> Residual<T> for Option<!> {
    type TryType = Option<T>;
}

impl<T, E> Try for Poll<Result<T, E>> {
    type Output = Poll<T>;
    type Residual = Result<!, E>;

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

impl<T, E, F: From<E>> FromResidual<Result<!, E>> for Poll<Result<T, F>> {
    #[inline]
    fn from_residual(x: Result<!, E>) -> Self {
        match x {
            Err(e) => Poll::Ready(Err(From::from(e))),
        }
    }
}

impl<T, E> Try for Poll<Option<Result<T, E>>> {
    type Output = Poll<Option<T>>;
    type Residual = Result<!, E>;

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

impl<T, E, F: From<E>> FromResidual<Result<!, E>> for Poll<Option<Result<T, F>>> {
    #[inline]
    fn from_residual(x: Result<!, E>) -> Self {
        match x {
            Err(e) => Poll::Ready(Some(Err(From::from(e)))),
        }
    }
}
