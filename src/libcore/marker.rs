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

#![stable(feature = "rust1", since = "1.0.0")]

use clone::Clone;
use cmp;
use default::Default;
use option::Option;
use hash::Hash;
use hash::Hasher;

/// Types that can be transferred across thread boundaries.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang = "send"]
#[rustc_on_unimplemented = "`{Self}` cannot be sent between threads safely"]
pub unsafe trait Send {
    // empty.
}

unsafe impl Send for .. { }

impl<T> !Send for *const T { }
impl<T> !Send for *mut T { }

/// Types with a constant size known at compile-time.
///
/// All type parameters which can be bounded have an implicit bound of `Sized`. The special syntax
/// `?Sized` can be used to remove this bound if it is not appropriate.
///
/// ```
/// # #![allow(dead_code)]
/// struct Foo<T>(T);
/// struct Bar<T: ?Sized>(T);
///
/// // struct FooUse(Foo<[i32]>); // error: Sized is not implemented for [i32]
/// struct BarUse(Bar<[i32]>); // OK
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[lang = "sized"]
#[rustc_on_unimplemented = "`{Self}` does not have a constant size known at compile-time"]
#[fundamental] // for Default, for example, which requires that `[T]: !Default` be evaluatable
pub trait Sized {
    // Empty.
}

/// Types that can be "unsized" to a dynamically sized type.
#[unstable(feature = "unsize", issue = "27732")]
#[lang="unsize"]
pub trait Unsize<T: ?Sized> {
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
/// # #[allow(dead_code)]
/// struct Point {
///    x: i32,
///    y: i32,
/// }
/// ```
///
/// A `struct` can be `Copy`, and `i32` is `Copy`, so therefore, `Point` is eligible to be `Copy`.
///
/// ```
/// # #![allow(dead_code)]
/// # struct Point;
/// struct PointList {
///     points: Vec<Point>,
/// }
/// ```
///
/// The `PointList` `struct` cannot implement `Copy`, because `Vec<T>` is not `Copy`. If we
/// attempt to derive a `Copy` implementation, we'll get an error:
///
/// ```text
/// the trait `Copy` may not be implemented for this type; field `points` does not implement `Copy`
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
///
/// # Derivable
///
/// This trait can be used with `#[derive]`.
#[stable(feature = "rust1", since = "1.0.0")]
#[lang = "copy"]
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
/// provide unsynchronized mutation. The trick is a mutable reference
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
/// Any types with interior mutability must also use the `std::cell::UnsafeCell`
/// wrapper around the value(s) which can be mutated when behind a `&`
/// reference; not doing this is undefined behavior (for example,
/// `transmute`-ing from `&T` to `&mut T` is invalid).
#[stable(feature = "rust1", since = "1.0.0")]
#[lang = "sync"]
#[rustc_on_unimplemented = "`{Self}` cannot be shared between threads safely"]
pub unsafe trait Sync {
    // Empty
}

unsafe impl Sync for .. { }

impl<T> !Sync for *const T { }
impl<T> !Sync for *mut T { }

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

        impl<T:?Sized> Default for $t<T> {
            fn default() -> $t<T> {
                $t
            }
        }
        )
}

/// `PhantomData<T>` allows you to describe that a type acts as if it stores a value of type `T`,
/// even though it does not. This allows you to inform the compiler about certain safety properties
/// of your code.
///
/// # A ghastly note 👻👻👻
///
/// Though they both have scary names, `PhantomData<T>` and 'phantom types' are related, but not
/// identical. Phantom types are a more general concept that don't require `PhantomData<T>` to
/// implement, but `PhantomData<T>` is the most common way to implement them in a correct manner.
///
/// # Examples
///
/// ## Unused lifetime parameter
///
/// Perhaps the most common time that `PhantomData` is required is
/// with a struct that has an unused lifetime parameter, typically as
/// part of some unsafe code. For example, here is a struct `Slice`
/// that has two pointers of type `*const T`, presumably pointing into
/// an array somewhere:
///
/// ```ignore
/// struct Slice<'a, T> {
///     start: *const T,
///     end: *const T,
/// }
/// ```
///
/// The intention is that the underlying data is only valid for the
/// lifetime `'a`, so `Slice` should not outlive `'a`. However, this
/// intent is not expressed in the code, since there are no uses of
/// the lifetime `'a` and hence it is not clear what data it applies
/// to. We can correct this by telling the compiler to act *as if* the
/// `Slice` struct contained a borrowed reference `&'a T`:
///
/// ```
/// use std::marker::PhantomData;
///
/// # #[allow(dead_code)]
/// struct Slice<'a, T:'a> {
///     start: *const T,
///     end: *const T,
///     phantom: PhantomData<&'a T>
/// }
/// ```
///
/// This also in turn requires that we annotate `T:'a`, indicating
/// that `T` is a type that can be borrowed for the lifetime `'a`.
///
/// ## Unused type parameters
///
/// It sometimes happens that there are unused type parameters that
/// indicate what type of data a struct is "tied" to, even though that
/// data is not actually found in the struct itself. Here is an
/// example where this arises when handling external resources over a
/// foreign function interface. `PhantomData<T>` can prevent
/// mismatches by enforcing types in the method implementations:
///
/// ```
/// # #![allow(dead_code)]
/// # trait ResType { fn foo(&self); }
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
/// ## Indicating ownership
///
/// Adding a field of type `PhantomData<T>` also indicates that your
/// struct owns data of type `T`. This in turn implies that when your
/// struct is dropped, it may in turn drop one or more instances of
/// the type `T`, though that may not be apparent from the other
/// structure of the type itself. This is commonly necessary if the
/// structure is using a raw pointer like `*mut T` whose referent
/// may be dropped when the type is dropped, as a `*mut T` is
/// otherwise not treated as owned.
///
/// If your struct does not in fact *own* the data of type `T`, it is
/// better to use a reference type, like `PhantomData<&'a T>`
/// (ideally) or `PhantomData<*const T>` (if no lifetime applies), so
/// as not to indicate ownership.
#[lang = "phantom_data"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct PhantomData<T:?Sized>;

impls! { PhantomData }

mod impls {
    use super::{Send, Sync, Sized};

    unsafe impl<'a, T: Sync + ?Sized> Send for &'a T {}
    unsafe impl<'a, T: Send + ?Sized> Send for &'a mut T {}
}

/// Types that can be reflected over.
///
/// This trait is implemented for all types. Its purpose is to ensure
/// that when you write a generic function that will employ
/// reflection, that must be reflected (no pun intended) in the
/// generic bounds of that function. Here is an example:
///
/// ```
/// #![feature(reflect_marker)]
/// use std::marker::Reflect;
/// use std::any::Any;
///
/// # #[allow(dead_code)]
/// fn foo<T:Reflect+'static>(x: &T) {
///     let any: &Any = x;
///     if any.is::<u32>() { println!("u32"); }
/// }
/// ```
///
/// Without the declaration `T:Reflect`, `foo` would not type check
/// (note: as a matter of style, it would be preferable to write
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
#[unstable(feature = "reflect_marker",
           reason = "requires RFC and more experience",
           issue = "27749")]
#[rustc_on_unimplemented = "`{Self}` does not implement `Any`; \
                            ensure all type parameters are bounded by `Any`"]
pub trait Reflect {}

impl Reflect for .. { }
