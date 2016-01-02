// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Panic support in the standard library

#![unstable(feature = "std_panic", reason = "awaiting feedback",
            issue = "27719")]

use cell::UnsafeCell;
use ops::{Deref, DerefMut};
use ptr::{Unique, Shared};
use rc::Rc;
use sync::{Arc, Mutex, RwLock};
use sys_common::unwind;
use thread::Result;

pub use panicking::{take_handler, set_handler, PanicInfo, Location};

/// A marker trait which represents "panic safe" types in Rust.
///
/// This trait is implemented by default for many types and behaves similarly in
/// terms of inference of implementation to the `Send` and `Sync` traits. The
/// purpose of this trait is to encode what types are safe to cross a `recover`
/// boundary with no fear of panic safety.
///
/// ## What is panic safety?
///
/// In Rust a function can "return" early if it either panics or calls a
/// function which transitively panics. This sort of control flow is not always
/// anticipated, and has the possibility of causing subtle bugs through a
/// combination of two cricial components:
///
/// 1. A data structure is in a temporarily invalid state when the thread
///    panics.
/// 2. This broken invariant is then later observed.
///
/// Typically in Rust, it is difficult to perform step (2) because catching a
/// panic involves either spawning a thread (which in turns makes it difficult
/// to later witness broken invariants) or using the `recover` function in this
/// module. Additionally, even if an invariant is witnessed, it typically isn't a
/// problem in Rust because there's no uninitialized values (like in C or C++).
///
/// It is possible, however, for **logical** invariants to be broken in Rust,
/// which can end up causing behavioral bugs. Another key aspect of panic safety
/// in Rust is that, in the absence of `unsafe` code, a panic cannot lead to
/// memory unsafety.
///
/// That was a bit of a whirlwind tour of panic safety, but for more information
/// about panic safety and how it applies to Rust, see an [associated RFC][rfc].
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// ## What is `RecoverSafe`?
///
/// Now that we've got an idea of what panic safety is in Rust, it's also
/// important to understand what this trait represents. As mentioned above, one
/// way to witness broken invariants is through the `recover` function in this
/// module as it allows catching a panic and then re-using the environment of
/// the closure.
///
/// Simply put, a type `T` implements `RecoverSafe` if it cannot easily allow
/// witnessing a broken invariant through the use of `recover` (catching a
/// panic). This trait is a marker trait, so it is automatically implemented for
/// many types, and it is also structurally composed (e.g. a struct is recover
/// safe if all of its components are recover safe).
///
/// Note, however, that this is not an unsafe trait, so there is not a succinct
/// contract that this trait is providing. Instead it is intended as more of a
/// "speed bump" to alert users of `recover` that broken invariants may be
/// witnessed and may need to be accounted for.
///
/// ## Who implements `RecoverSafe`?
///
/// Types such as `&mut T` and `&RefCell<T>` are examples which are **not**
/// recover safe. The general idea is that any mutable state which can be shared
/// across `recover` is not recover safe by default. This is because it is very
/// easy to witness a broken invariant outside of `recover` as the data is
/// simply accesed as usual.
///
/// Types like `&Mutex<T>`, however, are recover safe because they implement
/// poisoning by default. They still allow witnessing a broken invariant, but
/// they already provide their own "speed bumps" to do so.
///
/// ## When should `RecoverSafe` be used?
///
/// Is not intended that most types or functions need to worry about this trait.
/// It is only used as a bound on the `recover` function and as mentioned above,
/// the lack of `unsafe` means it is mostly an advisory. The `AssertRecoverSafe`
/// wrapper struct in this module can be used to force this trait to be
/// implemented for any closed over variables passed to the `recover` function
/// (more on this below).
#[unstable(feature = "recover", reason = "awaiting feedback", issue = "27719")]
#[rustc_on_unimplemented = "the type {Self} may not be safely transferred \
                            across a recover boundary"]
pub trait RecoverSafe {}

/// A marker trait representing types where a shared reference is considered
/// recover safe.
///
/// This trait is namely not implemented by `UnsafeCell`, the root of all
/// interior mutability.
///
/// This is a "helper marker trait" used to provide impl blocks for the
/// `RecoverSafe` trait, for more information see that documentation.
#[unstable(feature = "recover", reason = "awaiting feedback", issue = "27719")]
#[rustc_on_unimplemented = "the type {Self} contains interior mutability \
                            and a reference may not be safely transferrable \
                            across a recover boundary"]
pub trait RefRecoverSafe {}

/// A simple wrapper around a type to assert that it is panic safe.
///
/// When using `recover` it may be the case that some of the closed over
/// variables are not panic safe. For example if `&mut T` is captured the
/// compiler will generate a warning indicating that it is not panic safe. It
/// may not be the case, however, that this is actually a problem due to the
/// specific usage of `recover` if panic safety is specifically taken into
/// account. This wrapper struct is useful for a quick and lightweight
/// annotation that a variable is indeed panic safe.
///
/// # Examples
///
/// ```
/// #![feature(recover, std_panic)]
///
/// use std::panic::{self, AssertRecoverSafe};
///
/// let mut variable = 4;
///
/// // This code will not compile becuause the closure captures `&mut variable`
/// // which is not considered panic safe by default.
///
/// // panic::recover(|| {
/// //     variable += 3;
/// // });
///
/// // This, however, will compile due to the `AssertRecoverSafe` wrapper
/// let result = {
///     let mut wrapper = AssertRecoverSafe::new(&mut variable);
///     panic::recover(move || {
///         **wrapper += 3;
///     })
/// };
/// // ...
/// ```
#[unstable(feature = "recover", reason = "awaiting feedback", issue = "27719")]
pub struct AssertRecoverSafe<T>(T);

// Implementations of the `RecoverSafe` trait:
//
// * By default everything is recover safe
// * pointers T contains mutability of some form are not recover safe
// * Unique, an owning pointer, lifts an implementation
// * Types like Mutex/RwLock which are explicilty poisoned are recover safe
// * Our custom AssertRecoverSafe wrapper is indeed recover safe
impl RecoverSafe for .. {}
impl<'a, T: ?Sized> !RecoverSafe for &'a mut T {}
impl<'a, T: RefRecoverSafe + ?Sized> RecoverSafe for &'a T {}
impl<T: RefRecoverSafe + ?Sized> RecoverSafe for *const T {}
impl<T: RefRecoverSafe + ?Sized> RecoverSafe for *mut T {}
impl<T: RecoverSafe> RecoverSafe for Unique<T> {}
impl<T: RefRecoverSafe + ?Sized> RecoverSafe for Shared<T> {}
impl<T: ?Sized> RecoverSafe for Mutex<T> {}
impl<T: ?Sized> RecoverSafe for RwLock<T> {}
impl<T> RecoverSafe for AssertRecoverSafe<T> {}

// not covered via the Shared impl above b/c the inner contents use
// Cell/AtomicUsize, but the usage here is recover safe so we can lift the
// impl up one level to Arc/Rc itself
impl<T: RefRecoverSafe + ?Sized> RecoverSafe for Rc<T> {}
impl<T: RefRecoverSafe + ?Sized> RecoverSafe for Arc<T> {}

// Pretty simple implementations for the `RefRecoverSafe` marker trait,
// basically just saying that this is a marker trait and `UnsafeCell` is the
// only thing which doesn't implement it (which then transitively applies to
// everything else).
impl RefRecoverSafe for .. {}
impl<T: ?Sized> !RefRecoverSafe for UnsafeCell<T> {}
impl<T> RefRecoverSafe for AssertRecoverSafe<T> {}

impl<T> AssertRecoverSafe<T> {
    /// Creates a new `AssertRecoverSafe` wrapper around the provided type.
    #[unstable(feature = "recover", reason = "awaiting feedback", issue = "27719")]
    pub fn new(t: T) -> AssertRecoverSafe<T> {
        AssertRecoverSafe(t)
    }
}

impl<T> Deref for AssertRecoverSafe<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for AssertRecoverSafe<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// Invokes a closure, capturing the cause of panic if one occurs.
///
/// This function will return `Ok` with the closure's result if the closure
/// does not panic, and will return `Err(cause)` if the closure panics. The
/// `cause` returned is the object with which panic was originally invoked.
///
/// It is currently undefined behavior to unwind from Rust code into foreign
/// code, so this function is particularly useful when Rust is called from
/// another language (normally C). This can run arbitrary Rust code, capturing a
/// panic and allowing a graceful handling of the error.
///
/// It is **not** recommended to use this function for a general try/catch
/// mechanism. The `Result` type is more appropriate to use for functions that
/// can fail on a regular basis.
///
/// The closure provided is required to adhere to the `RecoverSafe` to ensure
/// that all captured variables are safe to cross this recover boundary. The
/// purpose of this bound is to encode the concept of [exception safety][rfc] in
/// the type system. Most usage of this function should not need to worry about
/// this bound as programs are naturally panic safe without `unsafe` code. If it
/// becomes a problem the associated `AssertRecoverSafe` wrapper type in this
/// module can be used to quickly assert that the usage here is indeed exception
/// safe.
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// # Examples
///
/// ```
/// #![feature(recover, std_panic)]
///
/// use std::panic;
///
/// let result = panic::recover(|| {
///     println!("hello!");
/// });
/// assert!(result.is_ok());
///
/// let result = panic::recover(|| {
///     panic!("oh no!");
/// });
/// assert!(result.is_err());
/// ```
#[unstable(feature = "recover", reason = "awaiting feedback", issue = "27719")]
pub fn recover<F: FnOnce() -> R + RecoverSafe, R>(f: F) -> Result<R> {
    let mut result = None;
    unsafe {
        let result = &mut result;
        try!(unwind::try(move || *result = Some(f())))
    }
    Ok(result.unwrap())
}
