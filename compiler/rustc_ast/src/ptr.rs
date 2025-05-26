//! The AST pointer.
//!
//! Provides [`P<T>`][struct@P], an owned smart pointer.
//!
//! # Motivations and benefits
//!
//! * **Identity**: sharing AST nodes is problematic for the various analysis
//!   passes (e.g., one may be able to bypass the borrow checker with a shared
//!   `ExprKind::AddrOf` node taking a mutable borrow).
//!
//! * **Efficiency**: folding can reuse allocation space for `P<T>` and `Vec<T>`,
//!   the latter even when the input and output types differ (as it would be the
//!   case with arenas or a GADT AST using type parameters to toggle features).
//!
//! * **Maintainability**: `P<T>` provides an interface, which can remain fully
//!   functional even if the implementation changes (using a special thread-local
//!   heap, for example). Moreover, a switch to, e.g., `P<'a, T>` would be easy
//!   and mostly automated.

use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut};

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

/// An owned smart pointer.
///
/// See the [module level documentation][crate::ptr] for details.
pub struct P<T: ?Sized> {
    ptr: Box<T>,
}

/// Construct a `P<T>` from a `T` value.
#[allow(non_snake_case)]
pub fn P<T>(value: T) -> P<T> {
    P { ptr: Box::new(value) }
}

impl<T> P<T> {
    /// Consume the `P` and return the wrapped value.
    pub fn into_inner(self) -> T {
        *self.ptr
    }
}

impl<T: ?Sized> Deref for P<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.ptr
    }
}

impl<T: ?Sized> DerefMut for P<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.ptr
    }
}

impl<T: Clone> Clone for P<T> {
    fn clone(&self) -> P<T> {
        P((**self).clone())
    }
}

impl<T: ?Sized + Debug> Debug for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.ptr, f)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for P<T> {
    fn decode(d: &mut D) -> P<T> {
        P(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for P<T> {
    fn encode(&self, s: &mut S) {
        (**self).encode(s);
    }
}
