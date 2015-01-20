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

#![stable]

use clone::Clone;

/// Types able to be transferred across task boundaries.
#[unstable = "will be overhauled with new lifetime rules; see RFC 458"]
#[lang="send"]
pub unsafe trait Send: 'static {
    // empty.
}

/// Types with a constant size known at compile-time.
#[stable]
#[lang="sized"]
pub trait Sized {
    // Empty.
}

/// Types that can be copied by simply copying bits (i.e. `memcpy`).
///
/// By default, variable bindings have 'move semantics.' In other
/// words:
///
/// ```
/// #[derive(Show)]
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
/// #[derive(Show, Copy)]
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
#[stable]
#[lang="copy"]
pub trait Copy {
    // Empty.
}

/// Types that can be safely shared between tasks when aliased.
///
/// The precise definition is: a type `T` is `Sync` if `&T` is
/// thread-safe. In other words, there is no possibility of data races
/// when passing `&T` references between tasks.
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
#[unstable = "will be overhauled with new lifetime rules; see RFC 458"]
#[lang="sync"]
pub unsafe trait Sync {
    // Empty
}


/// A marker type whose type parameter `T` is considered to be
/// covariant with respect to the type itself. This is (typically)
/// used to indicate that an instance of the type `T` is being stored
/// into memory and read from, even though that may not be apparent.
///
/// For more information about variance, refer to this Wikipedia
/// article <http://en.wikipedia.org/wiki/Variance_%28computer_science%29>.
///
/// *Note:* It is very unusual to have to add a covariant constraint.
/// If you are not sure, you probably want to use `InvariantType`.
///
/// # Example
///
/// Given a struct `S` that includes a type parameter `T`
/// but does not actually *reference* that type parameter:
///
/// ```ignore
/// use std::mem;
///
/// struct S<T> { x: *() }
/// fn get<T>(s: &S<T>) -> T {
///    unsafe {
///        let x: *T = mem::transmute(s.x);
///        *x
///    }
/// }
/// ```
///
/// The type system would currently infer that the value of
/// the type parameter `T` is irrelevant, and hence a `S<int>` is
/// a subtype of `S<Box<int>>` (or, for that matter, `S<U>` for
/// any `U`). But this is incorrect because `get()` converts the
/// `*()` into a `*T` and reads from it. Therefore, we should include the
/// a marker field `CovariantType<T>` to inform the type checker that
/// `S<T>` is a subtype of `S<U>` if `T` is a subtype of `U`
/// (for example, `S<&'static int>` is a subtype of `S<&'a int>`
/// for some lifetime `'a`, but not the other way around).
#[unstable = "likely to change with new variance strategy"]
#[lang="covariant_type"]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct CovariantType<T: ?Sized>;

impl<T: ?Sized> Copy for CovariantType<T> {}
impl<T: ?Sized> Clone for CovariantType<T> {
    fn clone(&self) -> CovariantType<T> { *self }
}

/// A marker type whose type parameter `T` is considered to be
/// contravariant with respect to the type itself. This is (typically)
/// used to indicate that an instance of the type `T` will be consumed
/// (but not read from), even though that may not be apparent.
///
/// For more information about variance, refer to this Wikipedia
/// article <http://en.wikipedia.org/wiki/Variance_%28computer_science%29>.
///
/// *Note:* It is very unusual to have to add a contravariant constraint.
/// If you are not sure, you probably want to use `InvariantType`.
///
/// # Example
///
/// Given a struct `S` that includes a type parameter `T`
/// but does not actually *reference* that type parameter:
///
/// ```
/// use std::mem;
///
/// struct S<T> { x: *const () }
/// fn get<T>(s: &S<T>, v: T) {
///    unsafe {
///        let x: fn(T) = mem::transmute(s.x);
///        x(v)
///    }
/// }
/// ```
///
/// The type system would currently infer that the value of
/// the type parameter `T` is irrelevant, and hence a `S<int>` is
/// a subtype of `S<Box<int>>` (or, for that matter, `S<U>` for
/// any `U`). But this is incorrect because `get()` converts the
/// `*()` into a `fn(T)` and then passes a value of type `T` to it.
///
/// Supplying a `ContravariantType` marker would correct the
/// problem, because it would mark `S` so that `S<T>` is only a
/// subtype of `S<U>` if `U` is a subtype of `T`; given that the
/// function requires arguments of type `T`, it must also accept
/// arguments of type `U`, hence such a conversion is safe.
#[unstable = "likely to change with new variance strategy"]
#[lang="contravariant_type"]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct ContravariantType<T: ?Sized>;

impl<T: ?Sized> Copy for ContravariantType<T> {}
impl<T: ?Sized> Clone for ContravariantType<T> {
    fn clone(&self) -> ContravariantType<T> { *self }
}

/// A marker type whose type parameter `T` is considered to be
/// invariant with respect to the type itself. This is (typically)
/// used to indicate that instances of the type `T` may be read or
/// written, even though that may not be apparent.
///
/// For more information about variance, refer to this Wikipedia
/// article <http://en.wikipedia.org/wiki/Variance_%28computer_science%29>.
///
/// # Example
///
/// The Cell type is an example which uses unsafe code to achieve
/// "interior" mutability:
///
/// ```
/// struct Cell<T> { value: T }
/// ```
///
/// The type system would infer that `value` is only read here and
/// never written, but in fact `Cell` uses unsafe code to achieve
/// interior mutability.
#[unstable = "likely to change with new variance strategy"]
#[lang="invariant_type"]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct InvariantType<T: ?Sized>;

#[unstable = "likely to change with new variance strategy"]
impl<T: ?Sized> Copy for InvariantType<T> {}
#[unstable = "likely to change with new variance strategy"]
impl<T: ?Sized> Clone for InvariantType<T> {
    fn clone(&self) -> InvariantType<T> { *self }
}

/// As `CovariantType`, but for lifetime parameters. Using
/// `CovariantLifetime<'a>` indicates that it is ok to substitute
/// a *longer* lifetime for `'a` than the one you originally
/// started with (e.g., you could convert any lifetime `'foo` to
/// `'static`). You almost certainly want `ContravariantLifetime`
/// instead, or possibly `InvariantLifetime`. The only case where
/// it would be appropriate is that you have a (type-casted, and
/// hence hidden from the type system) function pointer with a
/// signature like `fn(&'a T)` (and no other uses of `'a`). In
/// this case, it is ok to substitute a larger lifetime for `'a`
/// (e.g., `fn(&'static T)`), because the function is only
/// becoming more selective in terms of what it accepts as
/// argument.
///
/// For more information about variance, refer to this Wikipedia
/// article <http://en.wikipedia.org/wiki/Variance_%28computer_science%29>.
#[unstable = "likely to change with new variance strategy"]
#[lang="covariant_lifetime"]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CovariantLifetime<'a>;

/// As `ContravariantType`, but for lifetime parameters. Using
/// `ContravariantLifetime<'a>` indicates that it is ok to
/// substitute a *shorter* lifetime for `'a` than the one you
/// originally started with (e.g., you could convert `'static` to
/// any lifetime `'foo`). This is appropriate for cases where you
/// have an unsafe pointer that is actually a pointer into some
/// memory with lifetime `'a`, and thus you want to limit the
/// lifetime of your data structure to `'a`. An example of where
/// this is used is the iterator for vectors.
///
/// For more information about variance, refer to this Wikipedia
/// article <http://en.wikipedia.org/wiki/Variance_%28computer_science%29>.
#[unstable = "likely to change with new variance strategy"]
#[lang="contravariant_lifetime"]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContravariantLifetime<'a>;

/// As `InvariantType`, but for lifetime parameters. Using
/// `InvariantLifetime<'a>` indicates that it is not ok to
/// substitute any other lifetime for `'a` besides its original
/// value. This is appropriate for cases where you have an unsafe
/// pointer that is actually a pointer into memory with lifetime `'a`,
/// and this pointer is itself stored in an inherently mutable
/// location (such as a `Cell`).
#[unstable = "likely to change with new variance strategy"]
#[lang="invariant_lifetime"]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct InvariantLifetime<'a>;

/// A type which is considered "not POD", meaning that it is not
/// implicitly copyable. This is typically embedded in other types to
/// ensure that they are never copied, even if they lack a destructor.
#[unstable = "likely to change with new variance strategy"]
#[lang="no_copy_bound"]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_copy_implementations)]
pub struct NoCopy;

/// A type which is considered managed by the GC. This is typically
/// embedded in other types.
#[unstable = "likely to change with new variance strategy"]
#[lang="managed_bound"]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[allow(missing_copy_implementations)]
pub struct Managed;
