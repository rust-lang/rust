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
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="send"]
#[rustc_on_unimplemented = "`{Self}` cannot be sent between threads safely"]
#[allow(deprecated)]
pub unsafe trait Send : MarkerTrait {
    // empty.
}

unsafe impl Send for .. { }

impl<T> !Send for *const T { }
impl<T> !Send for *mut T { }
impl !Send for Managed { }

/// Types with a constant size known at compile-time.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="sized"]
#[rustc_on_unimplemented = "`{Self}` does not have a constant size known at compile-time"]
#[fundamental] // for Default, for example, which requires that `[T]: !Default` be evaluatable
#[allow(deprecated)]
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
/// #[derive(Debug, Copy, Clone)]
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
/// #[derive(Copy, Clone)]
/// struct MyStruct;
/// ```
///
/// and
///
/// ```
/// struct MyStruct;
/// impl Copy for MyStruct {}
/// impl Clone for MyStruct { fn clone(&self) -> MyStruct { *self } }
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
pub trait Copy : Clone {
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
/// Any types with interior mutability must also use the `std::cell::UnsafeCell` wrapper around the
/// value(s) which can be mutated when behind a `&` reference; not doing this is undefined
/// behaviour (for example, `transmute`-ing from `&T` to `&mut T` is illegal).
#[stable(feature = "rust1", since = "1.0.0")]
#[lang="sync"]
#[rustc_on_unimplemented = "`{Self}` cannot be shared between threads safely"]
#[allow(deprecated)]
pub unsafe trait Sync : MarkerTrait {
    // Empty
}

unsafe impl Sync for .. { }

impl<T> !Sync for *const T { }
impl<T> !Sync for *mut T { }
impl !Sync for Managed { }

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

/// `MarkerTrait` is deprecated and no longer needed.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "No longer needed")]
#[allow(deprecated)]
#[cfg(stage0)]
pub trait MarkerTrait : PhantomFn<Self,Self> { }

/// `MarkerTrait` is deprecated and no longer needed.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "No longer needed")]
#[allow(deprecated)]
#[cfg(not(stage0))]
pub trait MarkerTrait { }

#[allow(deprecated)]
impl<T:?Sized> MarkerTrait for T { }

/// `PhantomFn` is a deprecated marker trait that is no longer needed.
#[lang="phantom_fn"]
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "No longer needed")]
#[cfg(stage0)]
pub trait PhantomFn<A:?Sized,R:?Sized=()> {
}

/// `PhantomFn` is a deprecated marker trait that is no longer needed.
#[unstable(feature = "core", reason = "deprecated")]
#[deprecated(since = "1.0.0", reason = "No longer needed")]
#[cfg(not(stage0))]
pub trait PhantomFn<A:?Sized,R:?Sized=()> {
}

#[allow(deprecated)]
#[cfg(not(stage0))]
impl<A:?Sized,R:?Sized,T:?Sized> PhantomFn<A,R> for T { }

/// `PhantomData<T>` allows you to describe that a type acts as if it stores a value of type `T`,
/// even though it does not. This allows you to inform the compiler about certain safety properties
/// of your code.
///
/// Though they both have scary names, `PhantomData<T>` and "phantom types" are unrelated. 👻👻👻
///
/// # Examples
///
/// When handling external resources over a foreign function interface, `PhantomData<T>` can
/// prevent mismatches by enforcing types in the method implementations:
///
/// ```
/// # trait ResType { fn foo(&self); };
/// # struct ParamType;
/// # mod foreign_lib {
/// # pub fn new(_: usize) -> *mut () { 42 as *mut () }
/// # pub fn do_stuff(_: *mut (), _: usize) {}
/// # }
/// # fn convert_params(_: ParamType) -> usize { 42 }
/// use std::marker::PhantomData;
/// use std::mem;
///
/// struct ExternalResource<R> {
///    resource_handle: *mut (),
///    resource_type: PhantomData<R>,
/// }
///
/// impl<R: ResType> ExternalResource<R> {
///     fn new() -> ExternalResource<R> {
///         let size_of_res = mem::size_of::<R>();
///         ExternalResource {
///             resource_handle: foreign_lib::new(size_of_res),
///             resource_type: PhantomData,
///         }
///     }
///
///     fn do_stuff(&self, param: ParamType) {
///         let foreign_params = convert_params(param);
///         foreign_lib::do_stuff(self.resource_handle, foreign_params);
///     }
/// }
/// ```
///
/// Another example: embedding a `PhantomData<T>` will inform the compiler
/// that one or more instances of the type `T` could be dropped when
/// instances of the type itself is dropped, though that may not be
/// apparent from the other structure of the type itself. This is
/// commonly necessary if the structure is using an unsafe pointer
/// like `*mut T` whose referent may be dropped when the type is
/// dropped, as a `*mut T` is otherwise not treated as owned.
#[lang="phantom_data"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct PhantomData<T:?Sized>;

impls! { PhantomData }

mod impls {
    use super::{Send, Sync, Sized};

    unsafe impl<'a, T: Sync + ?Sized> Send for &'a T {}
    unsafe impl<'a, T: Send + ?Sized> Send for &'a mut T {}
}

/// A marker trait indicates a type that can be reflected over. This
/// trait is implemented for all types. Its purpose is to ensure that
/// when you write a generic function that will employ reflection,
/// that must be reflected (no pun intended) in the generic bounds of
/// that function. Here is an example:
///
/// ```
/// #![feature(core)]
/// use std::marker::Reflect;
/// use std::any::Any;
/// fn foo<T:Reflect+'static>(x: &T) {
///     let any: &Any = x;
///     if any.is::<u32>() { println!("u32"); }
/// }
/// ```
///
/// Without the declaration `T:Reflect`, `foo` would not type check
/// (note: as a matter of style, it would be preferable to to write
/// `T:Any`, because `T:Any` implies `T:Reflect` and `T:'static`, but
/// we use `Reflect` here to show how it works). The `Reflect` bound
/// thus serves to alert `foo`'s caller to the fact that `foo` may
/// behave differently depending on whether `T=u32` or not. In
/// particular, thanks to the `Reflect` bound, callers know that a
/// function declared like `fn bar<T>(...)` will always act in
/// precisely the same way no matter what type `T` is supplied,
/// because there are no bounds declared on `T`. (The ability for a
/// caller to reason about what a function may do based solely on what
/// generic bounds are declared is often called the ["parametricity
/// property"][1].)
///
/// [1]: http://en.wikipedia.org/wiki/Parametricity
#[rustc_reflect_like]
#[unstable(feature = "core", reason = "requires RFC and more experience")]
#[allow(deprecated)]
pub trait Reflect : MarkerTrait {
}

impl Reflect for .. { }

