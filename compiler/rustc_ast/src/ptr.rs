//! The AST pointer.
//!
//! Provides `P<T>`, a frozen owned smart pointer.
//!
//! # Motivations and benefits
//!
//! * **Identity**: sharing AST nodes is problematic for the various analysis
//!   passes (e.g., one may be able to bypass the borrow checker with a shared
//!   `ExprKind::AddrOf` node taking a mutable borrow).
//!
//! * **Immutability**: `P<T>` disallows mutating its inner `T`, unlike `Box<T>`
//!   (unless it contains an `Unsafe` interior, but that may be denied later).
//!   This mainly prevents mistakes, but can also enforces a kind of "purity".
//!
//! * **Efficiency**: folding can reuse allocation space for `P<T>` and `Vec<T>`,
//!   the latter even when the input and output types differ (as it would be the
//!   case with arenas or a GADT AST using type parameters to toggle features).
//!
//! * **Maintainability**: `P<T>` provides a fixed interface - `Deref`,
//!   `and_then` and `map` - which can remain fully functional even if the
//!   implementation changes (using a special thread-local heap, for example).
//!   Moreover, a switch to, e.g., `P<'a, T>` would be easy and mostly automated.

use std::fmt::{self, Debug, Display};
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};
use std::{slice, vec};

use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
/// An owned smart pointer.
pub struct P<T: ?Sized> {
    ptr: Box<T>,
}

/// Construct a `P<T>` from a `T` value.
#[allow(non_snake_case)]
pub fn P<T: 'static>(value: T) -> P<T> {
    P { ptr: Box::new(value) }
}

impl<T: 'static> P<T> {
    /// Move out of the pointer.
    /// Intended for chaining transformations not covered by `map`.
    pub fn and_then<U, F>(self, f: F) -> U
    where
        F: FnOnce(T) -> U,
    {
        f(*self.ptr)
    }

    /// Equivalent to `and_then(|x| x)`.
    pub fn into_inner(self) -> T {
        *self.ptr
    }

    /// Produce a new `P<T>` from `self` without reallocating.
    pub fn map<F>(mut self, f: F) -> P<T>
    where
        F: FnOnce(T) -> T,
    {
        let x = f(*self.ptr);
        *self.ptr = x;

        self
    }

    /// Optionally produce a new `P<T>` from `self` without reallocating.
    pub fn filter_map<F>(mut self, f: F) -> Option<P<T>>
    where
        F: FnOnce(T) -> Option<T>,
    {
        *self.ptr = f(*self.ptr)?;
        Some(self)
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

impl<T: 'static + Clone> Clone for P<T> {
    fn clone(&self) -> P<T> {
        P((**self).clone())
    }
}

impl<T: ?Sized + Debug> Debug for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.ptr, f)
    }
}

impl<T: Display> Display for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&**self, f)
    }
}

impl<T> fmt::Pointer for P<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&self.ptr, f)
    }
}

impl<D: Decoder, T: 'static + Decodable<D>> Decodable<D> for P<T> {
    fn decode(d: &mut D) -> P<T> {
        P(Decodable::decode(d))
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for P<T> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        (**self).encode(s)
    }
}

impl<T> P<[T]> {
    pub const fn new() -> P<[T]> {
        // HACK(eddyb) bypass the lack of a `const fn` to create an empty `Box<[T]>`
        // (as trait methods, `default` in this case, can't be `const fn` yet).
        P {
            ptr: unsafe {
                use std::ptr::NonNull;
                std::mem::transmute(NonNull::<[T; 0]>::dangling() as NonNull<[T]>)
            },
        }
    }

    #[inline(never)]
    pub fn from_vec(v: Vec<T>) -> P<[T]> {
        P { ptr: v.into_boxed_slice() }
    }

    #[inline(never)]
    pub fn into_vec(self) -> Vec<T> {
        self.ptr.into_vec()
    }
}

impl<T> Default for P<[T]> {
    /// Creates an empty `P<[T]>`.
    fn default() -> P<[T]> {
        P::new()
    }
}

impl<T: Clone> Clone for P<[T]> {
    fn clone(&self) -> P<[T]> {
        P::from_vec(self.to_vec())
    }
}

impl<T> From<Vec<T>> for P<[T]> {
    fn from(v: Vec<T>) -> Self {
        P::from_vec(v)
    }
}

impl<T> Into<Vec<T>> for P<[T]> {
    fn into(self) -> Vec<T> {
        self.into_vec()
    }
}

impl<T> FromIterator<T> for P<[T]> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> P<[T]> {
        P::from_vec(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for P<[T]> {
    type Item = T;
    type IntoIter = vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T> IntoIterator for &'a P<[T]> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.ptr.into_iter()
    }
}

impl<S: Encoder, T: Encodable<S>> Encodable<S> for P<[T]> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error> {
        Encodable::encode(&**self, s)
    }
}

impl<D: Decoder, T: Decodable<D>> Decodable<D> for P<[T]> {
    fn decode(d: &mut D) -> P<[T]> {
        P::from_vec(Decodable::decode(d))
    }
}

impl<CTX, T> HashStable<CTX> for P<T>
where
    T: ?Sized + HashStable<CTX>,
{
    fn hash_stable(&self, hcx: &mut CTX, hasher: &mut StableHasher) {
        (**self).hash_stable(hcx, hasher);
    }
}
