use crate::marker::Destruct;
use crate::ops::ControlFlow;

/// The `?` operator and `try {}` blocks.
///
/// `try_*` methods typically involve a type implementing this trait.  For
/// example, the closures passed to [`Iterator::try_fold`] and
/// [`Iterator::try_for_each`] must return such a type.
///
/// `Try` types are typically those containing two or more categories of values,
/// some subset of which are so commonly handled via early returns that it's
/// worth providing a terse (but still visible) syntax to make that easy.
///
/// This is most often seen for error handling with [`Result`] and [`Option`].
/// The quintessential implementation of this trait is on [`ControlFlow`].
///
/// # Using `Try` in Generic Code
///
/// `Iterator::try_fold` was stabilized to call back in Rust 1.27, but
/// this trait is much newer.  To illustrate the various associated types and
/// methods, let's implement our own version.
///
/// As a reminder, an infallible version of a fold looks something like this:
/// ```
/// fn simple_fold<A, T>(
///     iter: impl Iterator<Item = T>,
///     mut accum: A,
///     mut f: impl FnMut(A, T) -> A,
/// ) -> A {
///     for x in iter {
///         accum = f(accum, x);
///     }
///     accum
/// }
/// ```
///
/// So instead of `f` returning just an `A`, we'll need it to return some other
/// type that produces an `A` in the "don't short circuit" path.  Conveniently,
/// that's also the type we need to return from the function.
///
/// Let's add a new generic parameter `R` for that type, and bound it to the
/// output type that we want:
/// ```
/// # #![feature(try_trait_v2)]
/// # use std::ops::Try;
/// fn simple_try_fold_1<A, T, R: Try<Output = A>>(
///     iter: impl Iterator<Item = T>,
///     mut accum: A,
///     mut f: impl FnMut(A, T) -> R,
/// ) -> R {
///     todo!()
/// }
/// ```
///
/// If we get through the entire iterator, we need to wrap up the accumulator
/// into the return type using [`Try::from_output`]:
/// ```
/// # #![feature(try_trait_v2)]
/// # use std::ops::{ControlFlow, Try};
/// fn simple_try_fold_2<A, T, R: Try<Output = A>>(
///     iter: impl Iterator<Item = T>,
///     mut accum: A,
///     mut f: impl FnMut(A, T) -> R,
/// ) -> R {
///     for x in iter {
///         let cf = f(accum, x).branch();
///         match cf {
///             ControlFlow::Continue(a) => accum = a,
///             ControlFlow::Break(_) => todo!(),
///         }
///     }
///     R::from_output(accum)
/// }
/// ```
///
/// We'll also need [`FromResidual::from_residual`] to turn the residual back
/// into the original type.  But because it's a supertrait of `Try`, we don't
/// need to mention it in the bounds.  All types which implement `Try` can be
/// recreated from their corresponding residual, so we'll just call it:
/// ```
/// # #![feature(try_trait_v2)]
/// # use std::ops::{ControlFlow, Try};
/// pub fn simple_try_fold_3<A, T, R: Try<Output = A>>(
///     iter: impl Iterator<Item = T>,
///     mut accum: A,
///     mut f: impl FnMut(A, T) -> R,
/// ) -> R {
///     for x in iter {
///         let cf = f(accum, x).branch();
///         match cf {
///             ControlFlow::Continue(a) => accum = a,
///             ControlFlow::Break(r) => return R::from_residual(r),
///         }
///     }
///     R::from_output(accum)
/// }
/// ```
///
/// But this "call `branch`, then `match` on it, and `return` if it was a
/// `Break`" is exactly what happens inside the `?` operator.  So rather than
/// do all this manually, we can just use `?` instead:
/// ```
/// # #![feature(try_trait_v2)]
/// # use std::ops::Try;
/// fn simple_try_fold<A, T, R: Try<Output = A>>(
///     iter: impl Iterator<Item = T>,
///     mut accum: A,
///     mut f: impl FnMut(A, T) -> R,
/// ) -> R {
///     for x in iter {
///         accum = f(accum, x)?;
///     }
///     R::from_output(accum)
/// }
/// ```
#[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
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
#[doc(alias = "?")]
#[lang = "Try"]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait Try: [const] FromResidual<Self::Residual, Self::Kind> {
    /// A marker type to signify the general meaning of the `?` operation on this type.
    /// For example, `<Result as Try>::Kind = TryResult`, and `TryResult` should be used for
    /// custom "result-like" types which may contain an error.
    /// When using `?` within a function, the return type of the function must implement
    /// `FromResidual<T, <T as Try>::Kind>` where `T` is the type of the target of `?`.
    type Kind;

    /// The type of the value produced by `?` when *not* short-circuiting.
    #[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
    type Output;

    /// The type of the value passed to [`FromResidual::from_residual`]
    /// as part of `?` when short-circuiting.
    ///
    /// This represents the possible values of the `Self` type which are *not*
    /// represented by the `Output` type.
    ///
    /// # Note to Implementors
    ///
    /// The choice of this type is critical to interconversion.
    /// Unlike the `Output` type, which will often be a raw generic type,
    /// this type is typically a newtype of some sort to "color" the type
    /// so that it's distinguishable from the residuals of other types.
    ///
    /// This is why `Result<T, E>::Residual` is not `E`, but `Result<!, E>`.
    /// That way it's distinct from `ControlFlow<E>::Residual`, for example,
    /// and thus `?` on `ControlFlow` cannot be used in a method returning `Result`.
    ///
    /// If you're making a generic type `Foo<T>` that implements `Try<Output = T>`,
    /// then typically you can use `Foo<!>` as its `Residual`
    /// type: that type will have a "hole" in the correct place, and will maintain the
    /// "foo-ness" of the residual so other types need to opt-in to interconversion.
    #[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
    type Residual;

    /// Constructs the type from its `Output` type.
    ///
    /// This should be implemented consistently with the `branch` method
    /// such that applying the `?` operator will get back the original value:
    /// `Try::from_output(x).branch() --> ControlFlow::Continue(x)`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_trait_v2)]
    /// use std::ops::Try;
    ///
    /// assert_eq!(<Result<_, String> as Try>::from_output(3), Ok(3));
    /// assert_eq!(<Option<_> as Try>::from_output(4), Some(4));
    /// assert_eq!(
    ///     <std::ops::ControlFlow<String, _> as Try>::from_output(5),
    ///     std::ops::ControlFlow::Continue(5),
    /// );
    ///
    /// # fn make_question_mark_work() -> Option<()> {
    /// assert_eq!(Option::from_output(4)?, 4);
    /// # None }
    /// # make_question_mark_work();
    ///
    /// // This is used, for example, on the accumulator in `try_fold`:
    /// let r = std::iter::empty().try_fold(4, |_, ()| -> Option<_> { unreachable!() });
    /// assert_eq!(r, Some(4));
    /// ```
    #[lang = "from_output"]
    #[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
    fn from_output(output: Self::Output) -> Self;

    /// Used in `?` to decide whether the operator should produce a value
    /// (because this returned [`ControlFlow::Continue`])
    /// or propagate a value back to the caller
    /// (because this returned [`ControlFlow::Break`]).
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_trait_v2)]
    /// use std::ops::{ControlFlow, Try};
    ///
    /// assert_eq!(Ok::<_, String>(3).branch(), ControlFlow::Continue(3));
    /// assert_eq!(Err::<String, _>(3).branch(), ControlFlow::Break(Err(3)));
    ///
    /// assert_eq!(Some(3).branch(), ControlFlow::Continue(3));
    /// assert_eq!(None::<String>.branch(), ControlFlow::Break(None));
    ///
    /// assert_eq!(ControlFlow::<String, _>::Continue(3).branch(), ControlFlow::Continue(3));
    /// assert_eq!(
    ///     ControlFlow::<_, String>::Break(3).branch(),
    ///     ControlFlow::Break(ControlFlow::Break(3)),
    /// );
    /// ```
    #[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
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
            Kind = "core::option::TryOption",
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
            Kind = "core::result::TryResult",
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
            Kind = "core::ops::control_flow::TryControlFlow",
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
#[rustc_diagnostic_item = "FromResidual"]
#[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait FromResidual<R = <Self as Try>::Residual, Kind = <Self as Try>::Kind> {
    /// Constructs the type from a compatible `Residual` type.
    ///
    /// This should be implemented consistently with the `branch` method such
    /// that applying the `?` operator will get back an equivalent residual:
    /// `FromResidual::from_residual(r).branch() --> ControlFlow::Break(r)`.
    /// (The residual is not mandated to be *identical* when interconversion is involved.)
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(try_trait_v2)]
    /// use std::ops::{ControlFlow, FromResidual};
    ///
    /// assert_eq!(Result::<String, i64>::from_residual(Err(3_u8)), Err(3));
    /// assert_eq!(Option::<String>::from_residual(None), None);
    /// assert_eq!(
    ///     ControlFlow::<_, String>::from_residual(ControlFlow::Break(5)),
    ///     ControlFlow::Break(5),
    /// );
    /// ```
    #[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
    fn from_residual(residual: R) -> Self;
}

/// The `TryAs` trait is used to derive a type that is equivalent but with
/// `Try::Output` changed to `T`.
#[unstable(feature = "try_trait_v2", issue = "84277", old_name = "try_trait")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
pub const trait TryAs<T>: Try {
    /// The Try type that is similar to Self, but with `Output` changed to `T`
    type Try: [const] Try<Kind = Self::Kind, Output = T, Residual = Self::Residual>;
}

#[unstable(
    feature = "yeet_desugar_details",
    issue = "none",
    reason = "just here to simplify the desugaring; will never be stabilized"
)]
#[inline]
#[track_caller] // because `Result::from_residual` has it
#[lang = "from_yeet"]
#[allow(unreachable_pub)] // not-exposed but still used via lang-item
pub fn from_yeet<T, Y>(yeeted: Y) -> T
where
    T: Try<Residual: From<Y>>,
{
    T::from_residual(yeeted.into())
}

/// Used by `?` desugaring to call `Try::branch` and wrap the `Residual` with `TryResidual`.
#[unstable(feature = "try_trait_v2", issue = "84277")]
#[rustc_const_unstable(feature = "const_try", issue = "74935")]
#[expect(unreachable_pub)] // only reachable via lang items
#[inline] // FIXME: force would be nice, but fails -- see #148915
#[lang = "try_operator_branch"]
pub const fn try_operator_branch<T: [const] Try>(v: T) -> ControlFlow<TryResidual<T>, T::Output> {
    v.branch().map_break(TryResidual)
}

/// Used by `?` desugaring to wrap the `Residual` value and reference the target
/// `Try` type in a type parameter, purely for type checking.
#[unstable(feature = "try_internals", issue = "none")]
#[rustc_diagnostic_item = "TryResidual"]
#[expect(unreachable_pub)] // only reachable via lang items
#[repr(transparent)]
pub struct TryResidual<T: Try>(T::Residual)
// hack
where
    T::Residual: Sized;

impl<T: Try> TryResidual<T> {
    #[unstable(feature = "try_internals", issue = "none")]
    #[rustc_const_unstable(feature = "const_try", issue = "74935")]
    #[expect(unreachable_pub)] // only reachable via lang items
    #[lang = "into_try_heterogeneous"]
    #[inline]
    pub const fn into_try_heterogeneous<U>(self) -> U
    where
        U: [const] FromResidual<T::Residual, T::Kind>,
    {
        FromResidual::from_residual(self.0)
    }

    #[unstable(feature = "try_internals", issue = "none")]
    #[rustc_const_unstable(feature = "const_try", issue = "74935")]
    #[expect(unreachable_pub)] // only reachable via lang items
    #[lang = "into_try_homogeneous"]
    #[inline]
    pub const fn into_try_homogeneous<U>(self) -> <T as TryAs<U>>::Try
    where
        T: [const] TryAs<U>,
    {
        FromResidual::from_residual(self.0)
    }
}

/// An adapter for implementing non-try methods via the `Try` implementation.
///
/// Conceptually the same as `Result<T, !>`, but requiring less work in trait
/// solving and inhabited-ness checking and such, by being an obvious newtype
/// and not having `From` bounds lying around.
///
/// Not currently planned to be exposed publicly, so just `pub(crate)`.
#[repr(transparent)]
pub(crate) struct NeverShortCircuit<T>(pub T);

impl<T> NeverShortCircuit<T> {
    /// Wraps a unary function to produce one that wraps the output into a `NeverShortCircuit`.
    ///
    /// This is useful for implementing infallible functions in terms of the `try_` ones,
    /// without accidentally capturing extra generic parameters in a closure.
    #[inline]
    #[rustc_const_unstable(feature = "const_array", issue = "147606")]
    pub(crate) const fn wrap_mut_1<A, F>(
        mut f: F,
    ) -> impl [const] FnMut(A) -> Self + [const] Destruct
    where
        F: [const] FnMut(A) -> T + [const] Destruct,
    {
        const move |a| NeverShortCircuit(f(a))
    }

    #[inline]
    #[rustc_const_unstable(feature = "const_array", issue = "147606")]
    pub(crate) const fn wrap_mut_2<A, B, F>(
        mut f: F,
    ) -> impl [const] FnMut(A, B) -> Self + [const] Destruct
    where
        F: [const] FnMut(A, B) -> T + [const] Destruct,
    {
        const move |a, b| NeverShortCircuit(f(a, b))
    }
}

#[rustc_const_unstable(feature = "const_never_short_circuit", issue = "none")]
const impl<T, U> TryAs<U> for NeverShortCircuit<T> {
    type Try = NeverShortCircuit<U>;
}

pub(crate) struct TryNeverShortCircuit(());

#[rustc_const_unstable(feature = "const_never_short_circuit", issue = "none")]
const impl<T> Try for NeverShortCircuit<T> {
    type Kind = TryNeverShortCircuit;
    type Output = T;
    type Residual = !;

    #[inline]
    fn branch(self) -> ControlFlow<!, T> {
        ControlFlow::Continue(self.0)
    }

    #[inline]
    fn from_output(x: T) -> Self {
        NeverShortCircuit(x)
    }
}

#[rustc_const_unstable(feature = "const_never_short_circuit", issue = "none")]
const impl<T> FromResidual for NeverShortCircuit<T> {
    #[inline]
    fn from_residual(never: !) -> Self {
        match never {}
    }
}

/// Implement `FromResidual<Yeet<T>>` on your type to enable
/// `do yeet expr` syntax in functions returning your type.
#[unstable(feature = "try_trait_v2_yeet", issue = "96374")]
#[derive(Debug)]
pub struct Yeet<T>(pub T);
