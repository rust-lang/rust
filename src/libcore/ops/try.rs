// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// This trait has been superseded by the `Try` trait, but must remain
/// here as `?` is still lowered to it in stage0 .
#[cfg(stage0)]
#[unstable(feature = "question_mark_carrier", issue = "31436")]
pub trait Carrier {
    /// The type of the value when computation succeeds.
    type Success;
    /// The type of the value when computation errors out.
    type Error;

    /// Create a `Carrier` from a success value.
    fn from_success(_: Self::Success) -> Self;

    /// Create a `Carrier` from an error value.
    fn from_error(_: Self::Error) -> Self;

    /// Translate this `Carrier` to another implementation of `Carrier` with the
    /// same associated types.
    fn translate<T>(self) -> T where T: Carrier<Success=Self::Success, Error=Self::Error>;
}

#[cfg(stage0)]
#[unstable(feature = "question_mark_carrier", issue = "31436")]
impl<U, V> Carrier for Result<U, V> {
    type Success = U;
    type Error = V;

    fn from_success(u: U) -> Result<U, V> {
        Ok(u)
    }

    fn from_error(e: V) -> Result<U, V> {
        Err(e)
    }

    fn translate<T>(self) -> T
        where T: Carrier<Success=U, Error=V>
    {
        match self {
            Ok(u) => T::from_success(u),
            Err(e) => T::from_error(e),
        }
    }
}

struct _DummyErrorType;

impl Try for _DummyErrorType {
    type Ok = ();
    type Error = ();

    fn into_result(self) -> Result<Self::Ok, Self::Error> {
        Ok(())
    }

    fn from_ok(_: ()) -> _DummyErrorType {
        _DummyErrorType
    }

    fn from_error(_: ()) -> _DummyErrorType {
        _DummyErrorType
    }
}

/// A trait for customizing the behaviour of the `?` operator.
///
/// A type implementing `Try` is one that has a canonical way to view it
/// in terms of a success/failure dichotomy.  This trait allows both
/// extracting those success or failure values from an existing instance and
/// creating a new instance from a success or failure value.
#[unstable(feature = "try_trait", issue = "42327")]
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
