// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Primitive traits and marker types representing basic 'kinds' of types.
//!
//! Rust types can be classified in various useful ways according to
//! intrinsic properties of the type. These classifications, often called
//! 'kinds', are represented as traits.
//!
//! They cannot be implemented by user code, but are instead implemented
//! by the compiler automatically for the types to which they apply.
//!
//! Marker types are special types that are used with unsafe code to
//! inform the compiler of special constraints. Marker types should
//! only be needed when you are creating an abstraction that is
//! implemented using unsafe code. In that case, you may want to embed
//! some of the marker types below into your type.

#![stable(feature = "rust1", since = "1.0.0")]

use clone::Clone;
use cmp;
use option::Option;
use hash::Hash;
use hash::Hasher;

/// Types able to be transferred across thread boundaries.
#[unstable(feature = "core",
           reason = "will be overhauled with new lifetime rules; see RFC 458")]
#[lang="send"]
#[rustc_on_unimplemented = "`{Self}` cannot be sent between threads safely"]
#[cfg(stage0)]
pub unsafe trait Send: 'static {
    // empty.
}
/// Types able to be transferred across thread boundaries.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="send"]
#[rustc_on_unimplemented = "`{Self}` cannot be sent between threads safely"]
#[cfg(not(stage0))]
pub unsafe trait Send : MarkerTrait {
    // empty.
}

/// Types with a constant size known at compile-time.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="sized"]
#[rustc_on_unimplemented = "`{Self}` does not have a constant size known at compile-time"]
pub trait Sized : MarkerTrait {
    // Empty.
}

/// Types that can be copied by simply copying bits (i.e. `memcpy`).
///
/// By default, variable bindings have 'move semantics.' In other
/// words:
///
/// ```
/// #[derive(Debug)]
/// struct Foo;
///
/// let x = Foo;
///
/// let y = x;
///
/// // `x` has moved into `y`, and so cannot be used
///
/// // println!("{:?}", x); // error: use of moved value
/// ```
///
/// However, if a type implements `Copy`, it instead has 'copy semantics':
///
/// ```
/// // we can just derive a `Copy` implementation
/// #[derive(Debug, Copy)]
/// struct Foo;
///
/// let x = Foo;
///
/// let y = x;
///
/// // `y` is a copy of `x`
///
/// println!("{:?}", x); // A-OK!
/// ```
///
/// It's important to note that in these two examples, the only difference is if you are allowed to
/// access `x` after the assignment: a move is also a bitwise copy under the hood.
///
/// ## When can my type be `Copy`?
///
/// A type can implement `Copy` if all of its components implement `Copy`. For example, this
/// `struct` can be `Copy`:
///
/// ```
/// struct Point {
///    x: i32,
///    y: i32,
/// }
/// ```
///
/// A `struct` can be `Copy`, and `i32` is `Copy`, so therefore, `Point` is eligible to be `Copy`.
///
/// ```
/// # struct Point;
/// struct PointList {
///     points: Vec<Point>,
/// }
/// ```
///
/// The `PointList` `struct` cannot implement `Copy`, because `Vec<T>` is not `Copy`. If we
/// attempt to derive a `Copy` implementation, we'll get an error.
///
/// ```text
/// error: the trait `Copy` may not be implemented for this type; field `points` does not implement
/// `Copy`
/// ```
///
/// ## How can I implement `Copy`?
///
/// There are two ways to implement `Copy` on your type:
///
/// ```
/// #[derive(Copy)]
/// struct MyStruct;
/// ```
///
/// and
///
/// ```
/// struct MyStruct;
/// impl Copy for MyStruct {}
/// ```
///
/// There is a small difference between the two: the `derive` strategy will also place a `Copy`
/// bound on type parameters, which isn't always desired.
///
/// ## When can my type _not_ be `Copy`?
///
/// Some types can't be copied safely. For example, copying `&mut T` would create an aliased
/// mutable reference, and copying `String` would result in two attempts to free the same buffer.
///
/// Generalizing the latter case, any type implementing `Drop` can't be `Copy`, because it's
/// managing some resource besides its own `size_of::<T>()` bytes.
///
/// ## When should my type be `Copy`?
///
/// Generally speaking, if your type _can_ implement `Copy`, it should. There's one important thing
/// to consider though: if you think your type may _not_ be able to implement `Copy` in the future,
/// then it might be prudent to not implement `Copy`. This is because removing `Copy` is a breaking
/// change: that second example would fail to compile if we made `Foo` non-`Copy`.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="copy"]
pub trait Copy : MarkerTrait {
    // Empty.
}

/// Types that can be safely shared between threads when aliased.
///
/// The precise definition is: a type `T` is `Sync` if `&T` is
/// thread-safe. In other words, there is no possibility of data races
/// when passing `&T` references between threads.
///
/// As one would expect, primitive types like `u8` and `f64` are all
/// `Sync`, and so are simple aggregate types containing them (like
/// tuples, structs and enums). More instances of basic `Sync` types
/// include "immutable" types like `&T` and those with simple
/// inherited mutability, such as `Box<T>`, `Vec<T>` and most other
/// collection types. (Generic parameters need to be `Sync` for their
/// container to be `Sync`.)
///
/// A somewhat surprising consequence of the definition is `&mut T` is
/// `Sync` (if `T` is `Sync`) even though it seems that it might
/// provide unsynchronised mutation. The trick is a mutable reference
/// stored in an aliasable reference (that is, `& &mut T`) becomes
/// read-only, as if it were a `& &T`, hence there is no risk of a data
/// race.
///
/// Types that are not `Sync` are those that have "interior
/// mutability" in a non-thread-safe way, such as `Cell` and `RefCell`
/// in `std::cell`. These types allow for mutation of their contents
/// even when in an immutable, aliasable slot, e.g. the contents of
/// `&Cell<T>` can be `.set`, and do not ensure data races are
/// impossible, hence they cannot be `Sync`. A higher level example
/// of a non-`Sync` type is the reference counted pointer
/// `std::rc::Rc`, because any reference `&Rc<T>` can clone a new
/// reference, which modifies the reference counts in a non-atomic
/// way.
///
/// For cases when one does need thread-safe interior mutability,
/// types like the atomics in `std::sync` and `Mutex` & `RWLock` in
/// the `sync` crate do ensure that any mutation cannot cause data
/// races.  Hence these types are `Sync`.
///
/// Users writing their own types with interior mutability (or anything
/// else that is not thread-safe) should use the `NoSync` marker type
/// (from `std::marker`) to ensure that the compiler doesn't
/// consider the user-defined type to be `Sync`.  Any types with
/// interior mutability must also use the `std::cell::UnsafeCell` wrapper
/// around the value(s) which can be mutated when behind a `&`
/// reference; not doing this is undefined behaviour (for example,
/// `transmute`-ing from `&T` to `&mut T` is illegal).
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="sync"]
#[rustc_on_unimplemented = "`{Self}` cannot be shared between threads safely"]
pub unsafe trait Sync : MarkerTrait {
    // Empty
}

/// A type which is considered "not POD", meaning that it is not
/// implicitly copyable. This is typically embedded in other types to
/// ensure that they are never copied, even if they lack a destructor.
#[unstable(feature = "core",
           reason = "likely to change with new variance strategy")]
#[lang="no_copy_bound"]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NoCopy;

/// A type which is considered managed by the GC. This is typically
/// embedded in other types.
#[unstable(feature = "core",
           reason = "likely to change with new variance strategy")]
#[lang="managed_bound"]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Managed;

macro_rules! impls{
    ($t: ident) => (
        #[cfg(stage0)]
        impl<T:?Sized, S: Hasher> Hash<S> for $t<T> {
            #[inline]
            fn hash(&self, _: &mut S) {
            }
        }
        #[cfg(not(stage0))]
        impl<T:?Sized> Hash for $t<T> {
            #[inline]
            fn hash<H: Hasher>(&self, _: &mut H) {
            }
        }

        impl<T:?Sized> cmp::PartialEq for $t<T> {
            fn eq(&self, _other: &$t<T>) -> bool {
                true
            }
        }

        impl<T:?Sized> cmp::Eq for $t<T> {
        }

        impl<T:?Sized> cmp::PartialOrd for $t<T> {
            fn partial_cmp(&self, _other: &$t<T>) -> Option<cmp::Ordering> {
                Option::Some(cmp::Ordering::Equal)
            }
        }

        impl<T:?Sized> cmp::Ord for $t<T> {
            fn cmp(&self, _other: &$t<T>) -> cmp::Ordering {
                cmp::Ordering::Equal
            }
        }

        impl<T:?Sized> Copy for $t<T> { }

        impl<T:?Sized> Clone for $t<T> {
            fn clone(&self) -> $t<T> {
                $t
            }
        }
        )
}

/// `MarkerTrait` is intended to be used as the supertrait for traits
/// that don't have any methods but instead serve just to designate
/// categories of types. An example would be the `Send` trait, which
/// indicates types that are sendable: `Send` does not itself offer
/// any methods, but instead is used to gate access to data.
///
/// FIXME. Better documentation needed here!
pub trait MarkerTrait : PhantomFn<Self> { }
impl<T:?Sized> MarkerTrait for T { }

/// `PhantomFn` is a marker trait for use with traits that contain
/// type or lifetime parameters that do not appear in any of their
/// methods. In that case, you can either remove those parameters, or
/// add a `PhantomFn` supertrait that reflects the signature of
/// methods that compiler should "pretend" exists. This most commonly
/// occurs for traits with no methods: in that particular case, you
/// can extend `MarkerTrait`, which is equivalent to
/// `PhantomFn<Self>`.
///
/// # Example
///
/// As an example, consider a trait with no methods like `Even`, meant
/// to represent types that are "even":
///
/// ```rust,ignore
/// trait Even { }
/// ```
///
/// In this case, because the implicit parameter `Self` is unused, the
/// compiler will issue an error. The only purpose of this trait is to
/// categorize types (and hence instances of those types) as "even" or
/// not, so if we *were* going to have a method, it might look like:
///
/// ```rust,ignore
/// trait Even {
///     fn is_even(self) -> bool { true }
/// }
/// ```
///
/// Therefore, we can model a method like this as follows:
///
/// ```rust
/// use std::marker::PhantomFn;
/// trait Even : PhantomFn<Self> { }
/// ```
///
/// Another equivalent, but clearer, option would be to use
/// `MarkerTrait`:
///
/// ```rust
/// use std::marker::MarkerTrait;
/// trait Even : MarkerTrait { }
/// ```
///
/// # Parameters
///
/// - `A` represents the type of the method's argument. You can use a
///   tuple to represent "multiple" arguments. Any types appearing here
///   will be considered "contravariant".
/// - `R`, if supplied, represents the method's return type. This defaults
///   to `()` as it is rarely needed.
///
/// # Additional reading
///
/// More details and background can be found in [RFC 738][738].
///
/// [738]: https://github.com/rust-lang/rfcs/blob/master/text/0738-variance.md
#[lang="phantom_fn"]
#[stable(feature = "rust1", since = "1.0.0")]
pub trait PhantomFn<A:?Sized,R:?Sized=()> { }

#[cfg(stage0)] // built into the trait matching system after stage0
impl<A:?Sized, R:?Sized, U:?Sized> PhantomFn<A,R> for U { }

/// Specific to stage0. You should not be seeing these docs!
#[cfg(stage0)]
#[lang="covariant_type"] // only relevant to stage0
pub struct PhantomData<T:?Sized>;

/// `PhantomData` is a way to tell the compiler about fake fields.
/// Phantom data is required whenever type parameters are not used.
/// The idea is that if the compiler encounters a `PhantomData<T>`
/// instance, it will behave *as if* an instance of the type `T` were
/// present for the purpose of various automatic analyses.
///
/// For example, embedding a `PhantomData<T>` will inform the compiler
/// that one or more instances of the type `T` could be dropped when
/// instances of the type itself is dropped, though that may not be
/// apparent from the other structure of the type itself. This is
/// commonly necessary if the structure is using an unsafe pointer
/// like `*mut T` whose referent may be dropped when the type is
/// dropped, as a `*mut T` is otherwise not treated as owned.
///
/// FIXME. Better documentation and examples of common patterns needed
/// here! For now, please see [RFC 738][738] for more information.
///
/// [738]: https://github.com/rust-lang/rfcs/blob/master/text/0738-variance.md
#[cfg(not(stage0))]
#[lang="phantom_data"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct PhantomData<T:?Sized>;

impls! { PhantomData }

#[cfg(not(stage0))]
mod impls {
    use super::{Send, Sync, Sized};

    unsafe impl<'a, T: Sync + ?Sized> Send for &'a T {}
    unsafe impl<'a, T: Send + ?Sized> Send for &'a mut T {}
}

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<&'a ()>`")]
#[lang="contravariant_lifetime"]
pub struct ContravariantLifetime<'a>;

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<fn(&'a ())>`")]
#[lang="covariant_lifetime"]
pub struct CovariantLifetime<'a>;

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<Cell<&'a ()>>`")]
#[lang="invariant_lifetime"]
pub struct InvariantLifetime<'a>;

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<fn(T)>`")]
#[lang="contravariant_type"]
pub struct ContravariantType<T>;

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<T>`")]
#[lang="covariant_type"]
#[cfg(not(stage0))]
pub struct CovariantType<T>;

/// Old-style marker trait. Deprecated.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "Replace with `PhantomData<Cell<T>>`")]
#[lang="invariant_type"]
pub struct InvariantType<T>;
