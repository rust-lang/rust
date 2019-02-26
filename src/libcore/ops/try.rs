/// A trait for customizing the behavior of the `?` operator.
///
/// A type implementing `Try` is one that has a canonical way to view it
/// in terms of a success/failure dichotomy. This trait allows both
/// extracting those success or failure values from an existing instance and
/// creating a new instance from a success or failure value.
#[unstable(feature = "try_trait", issue = "42327")]
#[rustc_on_unimplemented(
   on(all(
       any(from_method="from_error", from_method="from_ok"),
       from_desugaring="?"),
      message="the `?` operator can only be used in a \
               function that returns `Result` or `Option` \
               (or another type that implements `{Try}`)",
      label="cannot use the `?` operator in a function that returns `{Self}`"),
   on(all(from_method="into_result", from_desugaring="?"),
      message="the `?` operator can only be applied to values \
               that implement `{Try}`",
      label="the `?` operator cannot be applied to type `{Self}`")
)]
#[doc(alias = "?")]
pub trait Try {
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
    #[unstable(feature = "try_trait", issue = "42327")]
    fn into_result(self) -> Result<Self::Ok, Self::Error>;

    /// Wrap an error value to construct the composite result. For example,
    /// `Result::Err(x)` and `Result::from_error(x)` are equivalent.
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_error(v: Self::Error) -> Self;

    /// Wrap an OK value to construct the composite result. For example,
    /// `Result::Ok(x)` and `Result::from_ok(x)` are equivalent.
    #[unstable(feature = "try_trait", issue = "42327")]
    fn from_ok(v: Self::Ok) -> Self;
}
