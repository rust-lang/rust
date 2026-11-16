//! # Place Wrappers
//!
//! Place wrappers are a special kind of place proxy. They "physically contain"
//! the place they are proxying for. A good example is [`MaybeUninit<T>`].
//!
//! To support subplaces of these place wrappers, the [`WrapPlace`] trait
//! exists. It allows forwarding subplaces to the proxy and changing the
//! subplace access information. With [`MaybeUninit<T>`], this allows accessing
//! any subplace under the transformation that its type is wrapped in
//! `MaybeUninit`. So given `&MaybeUninit<Struct>`, the `field` subplace can be
//! borrowed using `&` and it has type `&MaybeUninit<Field>`.
//!
//! [`MaybeUninit<T>`]: std::mem::MaybeUninit

use crate::field::Subplace;

/// Wrap a place, exposing some of its subplaces.
pub trait PlaceWrapper {
    type Inner: ?Sized;
}

/// Expose a modified subplace of the wrapped place -- `place.field`.
pub unsafe trait WrapPlace<S>: PlaceWrapper
where
    S: Subplace<Source = Self::Inner>,
{
    type Wrapped: Subplace<Source = Self>;

    fn wrap(subplace: S) -> Self::Wrapped;
}
