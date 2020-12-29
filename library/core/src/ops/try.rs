use super::ControlFlow;

/// A trait for customizing the behavior of the `?` operator.
///
/// A type implementing `Try` is one that has a canonical way to view it
/// in terms of a success/failure dichotomy. This trait allows both
/// extracting those success or failure values from an existing instance and
/// creating a new instance from a success or failure value.
#[unstable(feature = "try_trait", issue = "42327")]
#[rustc_on_unimplemented(
    on(
        all(
            any(from_method = "from_error", from_method = "from_ok"),
            from_desugaring = "QuestionMark"
        ),
        message = "the `?` operator can only be used in {ItemContext} \
                    that returns `Result` or `Option` \
                    (or another type that implements `{Try2015}`)",
        label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
        enclosing_scope = "this function should return `Result` or `Option` to accept `?`"
    ),
    on(
        all(from_method = "into_result", from_desugaring = "QuestionMark"),
        message = "the `?` operator can only be applied to values \
                    that implement `{Try2015}`",
        label = "the `?` operator cannot be applied to type `{Self}`"
    )
)]
#[doc(alias = "?")]
#[lang = "try"]
pub trait Try2015 {
    /// The type of this value when viewed as successful.
    #[unstable(feature = "try_trait", issue = "42327")]
    type Ok;
    /// The type of this value when viewed as failed.
    #[unstable(feature = "try_trait", issue = "42327")]
    type Error;

    /// Applies the "?" operator. A return of `Ok(t)` means that the
    /// execution should continue normally, and the result of `?` is the
    /// value `t`. A return of `Err(e)` means that execution should branch
    /// to the innermost enclosing `catch`, or return from the function.
    ///
    /// If an `Err(e)` result is returned, the value `e` will be "wrapped"
    /// in the return type of the enclosing scope (which must itself implement
    /// `Try`). Specifically, the value `X::from_error(From::from(e))`
    /// is returned, where `X` is the return type of the enclosing function.
    #[lang = "into_result"]
    #[unstable(feature = "try_trait", issue = "42327")]
    fn into_result(self) -> Result<Self::Ok, Self::Error>;

    /// Wrap an error value to construct the composite result. For example,
    /// `Result::Err(x)` and `Result::from_error(x)` are equivalent.
    #[lang = "from_error"]
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_error(v: Self::Error) -> Self;

    /// Wrap an OK value to construct the composite result. For example,
    /// `Result::Ok(x)` and `Result::from_ok(x)` are equivalent.
    #[lang = "from_ok"]
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_ok(v: Self::Ok) -> Self;
}

/// A trait for customizing the behavior of the `?` operator.
///
/// This trait contains the behaviour core to this type,
/// such as the associated `Continue` type that both
/// is produced by the `?` operator and
/// is expected by a `try{}` block producing this type.
#[rustc_on_unimplemented(
    on(
        all(from_method = "continue_with", from_desugaring = "QuestionMark"),
        message = "the `?` operator can only be used in {ItemContext} \
                    that returns `Result` or `Option` \
                    (or another type that implements `{Bubble}`)",
        label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
        enclosing_scope = "this function should return `Result` or `Option` to accept `?`"
    ),
    on(
        all(from_method = "branch", from_desugaring = "QuestionMark"),
        message = "the `?` operator can only be applied to values \
                    that implement `{Bubble}`",
        label = "the `?` operator cannot be applied to type `{Self}`"
    )
)]
#[unstable(feature = "try_trait_v2", issue = "42327")]
pub trait Bubble {
    /// The type of the value consumed or produced when not short-circuiting.
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    // Temporarily using `Ok` still so I don't need to change the bounds in the library
    //type Continue;
    type Ok;

    /// A type that "colours" the short-circuit value so it can stay associated
    /// with the type constructor from which it came.
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    // This could have required that we can get back here via the holder,
    // but that means that every type needs a distinct holder.  It was removed
    // so that the Poll impls can use Result's Holder.
    //type Holder: BreakHolder<Self::Continue, Output = Self>;
    type Holder: BreakHolder<Self::Ok>;

    /// Used in `try{}` blocks to wrap the result of the block.
    #[cfg_attr(not(bootstrap), lang = "continue_with")]
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    fn continue_with(x: Self::Ok) -> Self;

    /// Determine whether to short-circuit (by returning `ControlFlow::Break`)
    /// or continue executing (by returning `ControlFlow::Continue`).
    #[cfg_attr(not(bootstrap), lang = "branch")]
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    fn branch(self) -> ControlFlow<Self::Holder, Self::Ok>;

    /// Demonstration that this is usable for different-return-type scenarios (like `Iterator::try_find`).
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    fn map<T>(self, f: impl FnOnce(Self::Ok) -> T) -> <Self::Holder as BreakHolder<T>>::Output
    where
        Self: Try2021,
        Self::Holder: BreakHolder<T>,
    {
        match Bubble::branch(self) {
            ControlFlow::Continue(c) => Bubble::continue_with(f(c)),
            //ControlFlow::Break(h) => BreakHolder::<T>::expand(h),
            ControlFlow::Break(h) => Try2021::from_holder(h),
        }
    }
}

/// The bound on a `<T as Try>::Holder` type that allows getting back to the original.
#[unstable(feature = "try_trait_v2", issue = "42327")]
pub trait BreakHolder<T>: Sized {
    /// The type from the original type constructor that also has this holder type,
    /// but has the specified Continue type.
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    type Output: Try2021<Ok = T, Holder = Self>;

    /* Superfluous now that `FromHolder` exists
    /// Rebuild the associated `impl Try` type from this holder.
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    fn expand(x: Self) -> Self::Output;
    */
}

/// Allows you to pick with other types can be converted into your `Try` type.
///
/// With the default type argument, this functions as a bound for a "normal"
/// `Try` type: one that can be split apart and put back together in either way.
///
/// For more complicated scenarios you'll likely need to bound on more than just this.
#[rustc_on_unimplemented(on(
    all(from_method = "from_holder", from_desugaring = "QuestionMark"),
    message = "the `?` operator can only be used in {ItemContext} \
                    that returns `Result` or `Option` \
                    (or another type that implements `{Try2021}`)",
    label = "cannot use the `?` operator in {ItemContext} that returns `{Self}`",
    enclosing_scope = "this function should return `Result` or `Option` to accept `?`"
))]
#[unstable(feature = "try_trait_v2", issue = "42327")]
pub trait Try2021<T = <Self as Bubble>::Holder>: Bubble
where
    T: BreakHolder<Self::Ok>,
{
    /// Perform conversion on this holder
    #[cfg_attr(not(bootstrap), lang = "from_holder")]
    #[unstable(feature = "try_trait_v2", issue = "42327")]
    fn from_holder(x: T) -> Self;
}
