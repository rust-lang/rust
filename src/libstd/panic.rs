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

#![stable(feature = "std_panic", since = "1.9.0")]

use any::Any;
use cell::UnsafeCell;
use fmt;
use ops::{Deref, DerefMut};
use panicking;
use ptr::{Unique, Shared};
use rc::Rc;
use sync::{Arc, Mutex, RwLock, atomic};
use thread::Result;

#[stable(feature = "panic_hooks", since = "1.10.0")]
pub use panicking::{take_hook, set_hook, PanicInfo, Location};

/// A marker trait which represents "panic safe" types in Rust.
///
/// This trait is implemented by default for many types and behaves similarly in
/// terms of inference of implementation to the `Send` and `Sync` traits. The
/// purpose of this trait is to encode what types are safe to cross a `catch_unwind`
/// boundary with no fear of unwind safety.
///
/// ## What is unwind safety?
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
/// to later witness broken invariants) or using the `catch_unwind` function in this
/// module. Additionally, even if an invariant is witnessed, it typically isn't a
/// problem in Rust because there are no uninitialized values (like in C or C++).
///
/// It is possible, however, for **logical** invariants to be broken in Rust,
/// which can end up causing behavioral bugs. Another key aspect of unwind safety
/// in Rust is that, in the absence of `unsafe` code, a panic cannot lead to
/// memory unsafety.
///
/// That was a bit of a whirlwind tour of unwind safety, but for more information
/// about unwind safety and how it applies to Rust, see an [associated RFC][rfc].
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// ## What is `UnwindSafe`?
///
/// Now that we've got an idea of what unwind safety is in Rust, it's also
/// important to understand what this trait represents. As mentioned above, one
/// way to witness broken invariants is through the `catch_unwind` function in this
/// module as it allows catching a panic and then re-using the environment of
/// the closure.
///
/// Simply put, a type `T` implements `UnwindSafe` if it cannot easily allow
/// witnessing a broken invariant through the use of `catch_unwind` (catching a
/// panic). This trait is a marker trait, so it is automatically implemented for
/// many types, and it is also structurally composed (e.g. a struct is unwind
/// safe if all of its components are unwind safe).
///
/// Note, however, that this is not an unsafe trait, so there is not a succinct
/// contract that this trait is providing. Instead it is intended as more of a
/// "speed bump" to alert users of `catch_unwind` that broken invariants may be
/// witnessed and may need to be accounted for.
///
/// ## Who implements `UnwindSafe`?
///
/// Types such as `&mut T` and `&RefCell<T>` are examples which are **not**
/// unwind safe. The general idea is that any mutable state which can be shared
/// across `catch_unwind` is not unwind safe by default. This is because it is very
/// easy to witness a broken invariant outside of `catch_unwind` as the data is
/// simply accessed as usual.
///
/// Types like `&Mutex<T>`, however, are unwind safe because they implement
/// poisoning by default. They still allow witnessing a broken invariant, but
/// they already provide their own "speed bumps" to do so.
///
/// ## When should `UnwindSafe` be used?
///
/// Is not intended that most types or functions need to worry about this trait.
/// It is only used as a bound on the `catch_unwind` function and as mentioned above,
/// the lack of `unsafe` means it is mostly an advisory. The `AssertUnwindSafe`
/// wrapper struct in this module can be used to force this trait to be
/// implemented for any closed over variables passed to the `catch_unwind` function
/// (more on this below).
#[stable(feature = "catch_unwind", since = "1.9.0")]
#[rustc_on_unimplemented = "the type {Self} may not be safely transferred \
                            across an unwind boundary"]
pub trait UnwindSafe {}

/// A marker trait representing types where a shared reference is considered
/// unwind safe.
///
/// This trait is namely not implemented by `UnsafeCell`, the root of all
/// interior mutability.
///
/// This is a "helper marker trait" used to provide impl blocks for the
/// `UnwindSafe` trait, for more information see that documentation.
#[stable(feature = "catch_unwind", since = "1.9.0")]
#[rustc_on_unimplemented = "the type {Self} contains interior mutability \
                            and a reference may not be safely transferrable \
                            across a catch_unwind boundary"]
pub trait RefUnwindSafe {}

/// A simple wrapper around a type to assert that it is unwind safe.
///
/// When using `catch_unwind` it may be the case that some of the closed over
/// variables are not unwind safe. For example if `&mut T` is captured the
/// compiler will generate a warning indicating that it is not unwind safe. It
/// may not be the case, however, that this is actually a problem due to the
/// specific usage of `catch_unwind` if unwind safety is specifically taken into
/// account. This wrapper struct is useful for a quick and lightweight
/// annotation that a variable is indeed unwind safe.
///
/// # Examples
///
/// One way to use `AssertUnwindSafe` is to assert that the entire closure
/// itself is unwind safe, bypassing all checks for all variables:
///
/// ```
/// use std::panic::{self, AssertUnwindSafe};
///
/// let mut variable = 4;
///
/// // This code will not compile because the closure captures `&mut variable`
/// // which is not considered unwind safe by default.
///
/// // panic::catch_unwind(|| {
/// //     variable += 3;
/// // });
///
/// // This, however, will compile due to the `AssertUnwindSafe` wrapper
/// let result = panic::catch_unwind(AssertUnwindSafe(|| {
///     variable += 3;
/// }));
/// // ...
/// ```
///
/// Wrapping the entire closure amounts to a blanket assertion that all captured
/// variables are unwind safe. This has the downside that if new captures are
/// added in the future, they will also be considered unwind safe. Therefore,
/// you may prefer to just wrap individual captures, as shown below. This is
/// more annotation, but it ensures that if a new capture is added which is not
/// unwind safe, you will get a compilation error at that time, which will
/// allow you to consider whether that new capture in fact represent a bug or
/// not.
///
/// ```
/// use std::panic::{self, AssertUnwindSafe};
///
/// let mut variable = 4;
/// let other_capture = 3;
///
/// let result = {
///     let mut wrapper = AssertUnwindSafe(&mut variable);
///     panic::catch_unwind(move || {
///         **wrapper += other_capture;
///     })
/// };
/// // ...
/// ```
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub struct AssertUnwindSafe<T>(
    #[stable(feature = "catch_unwind", since = "1.9.0")]
    pub T
);

// Implementations of the `UnwindSafe` trait:
//
// * By default everything is unwind safe
// * pointers T contains mutability of some form are not unwind safe
// * Unique, an owning pointer, lifts an implementation
// * Types like Mutex/RwLock which are explicilty poisoned are unwind safe
// * Our custom AssertUnwindSafe wrapper is indeed unwind safe
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl UnwindSafe for .. {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<'a, T: ?Sized> !UnwindSafe for &'a mut T {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<'a, T: RefUnwindSafe + ?Sized> UnwindSafe for &'a T {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for *const T {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for *mut T {}
#[unstable(feature = "unique", issue = "27730")]
impl<T: UnwindSafe> UnwindSafe for Unique<T> {}
#[unstable(feature = "shared", issue = "27730")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Shared<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: ?Sized> UnwindSafe for Mutex<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: ?Sized> UnwindSafe for RwLock<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T> UnwindSafe for AssertUnwindSafe<T> {}

// not covered via the Shared impl above b/c the inner contents use
// Cell/AtomicUsize, but the usage here is unwind safe so we can lift the
// impl up one level to Arc/Rc itself
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Rc<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Arc<T> {}

// Pretty simple implementations for the `RefUnwindSafe` marker trait,
// basically just saying that this is a marker trait and `UnsafeCell` is the
// only thing which doesn't implement it (which then transitively applies to
// everything else).
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl RefUnwindSafe for .. {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T: ?Sized> !RefUnwindSafe for UnsafeCell<T> {}
#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T> RefUnwindSafe for AssertUnwindSafe<T> {}

#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for Mutex<T> {}
#[stable(feature = "unwind_safe_lock_refs", since = "1.12.0")]
impl<T: ?Sized> RefUnwindSafe for RwLock<T> {}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "unwind_safe_atomic_refs", since = "1.14.0")]
impl RefUnwindSafe for atomic::AtomicIsize {}
#[cfg(target_has_atomic = "8")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicI8 {}
#[cfg(target_has_atomic = "16")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicI16 {}
#[cfg(target_has_atomic = "32")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicI32 {}
#[cfg(target_has_atomic = "64")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicI64 {}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "unwind_safe_atomic_refs", since = "1.14.0")]
impl RefUnwindSafe for atomic::AtomicUsize {}
#[cfg(target_has_atomic = "8")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicU8 {}
#[cfg(target_has_atomic = "16")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicU16 {}
#[cfg(target_has_atomic = "32")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicU32 {}
#[cfg(target_has_atomic = "64")]
#[unstable(feature = "integer_atomics", issue = "32976")]
impl RefUnwindSafe for atomic::AtomicU64 {}

#[cfg(target_has_atomic = "8")]
#[stable(feature = "unwind_safe_atomic_refs", since = "1.14.0")]
impl RefUnwindSafe for atomic::AtomicBool {}

#[cfg(target_has_atomic = "ptr")]
#[stable(feature = "unwind_safe_atomic_refs", since = "1.14.0")]
impl<T> RefUnwindSafe for atomic::AtomicPtr<T> {}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T> Deref for AssertUnwindSafe<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<T> DerefMut for AssertUnwindSafe<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

#[stable(feature = "catch_unwind", since = "1.9.0")]
impl<R, F: FnOnce() -> R> FnOnce<()> for AssertUnwindSafe<F> {
    type Output = R;

    extern "rust-call" fn call_once(self, _args: ()) -> R {
        (self.0)()
    }
}

#[stable(feature = "std_debug", since = "1.15.0")]
impl<T: fmt::Debug> fmt::Debug for AssertUnwindSafe<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("AssertUnwindSafe")
            .field(&self.0)
            .finish()
    }
}

/// Invokes a closure, capturing the cause of an unwinding panic if one occurs.
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
/// can fail on a regular basis. Additionally, this function is not guaranteed
/// to catch all panics, see the "Notes" section below.
///
/// The closure provided is required to adhere to the `UnwindSafe` trait to ensure
/// that all captured variables are safe to cross this boundary. The purpose of
/// this bound is to encode the concept of [exception safety][rfc] in the type
/// system. Most usage of this function should not need to worry about this
/// bound as programs are naturally unwind safe without `unsafe` code. If it
/// becomes a problem the associated `AssertUnwindSafe` wrapper type in this
/// module can be used to quickly assert that the usage here is indeed unwind
/// safe.
///
/// [rfc]: https://github.com/rust-lang/rfcs/blob/master/text/1236-stabilize-catch-panic.md
///
/// # Notes
///
/// Note that this function **may not catch all panics** in Rust. A panic in
/// Rust is not always implemented via unwinding, but can be implemented by
/// aborting the process as well. This function *only* catches unwinding panics,
/// not those that abort the process.
///
/// # Examples
///
/// ```
/// use std::panic;
///
/// let result = panic::catch_unwind(|| {
///     println!("hello!");
/// });
/// assert!(result.is_ok());
///
/// let result = panic::catch_unwind(|| {
///     panic!("oh no!");
/// });
/// assert!(result.is_err());
/// ```
#[stable(feature = "catch_unwind", since = "1.9.0")]
pub fn catch_unwind<F: FnOnce() -> R + UnwindSafe, R>(f: F) -> Result<R> {
    unsafe {
        panicking::try(f)
    }
}

/// Triggers a panic without invoking the panic hook.
///
/// This is designed to be used in conjunction with `catch_unwind` to, for
/// example, carry a panic across a layer of C code.
///
/// # Notes
///
/// Note that panics in Rust are not always implemented via unwinding, but they
/// may be implemented by aborting the process. If this function is called when
/// panics are implemented this way then this function will abort the process,
/// not trigger an unwind.
///
/// # Examples
///
/// ```should_panic
/// use std::panic;
///
/// let result = panic::catch_unwind(|| {
///     panic!("oh no!");
/// });
///
/// if let Err(err) = result {
///     panic::resume_unwind(err);
/// }
/// ```
#[stable(feature = "resume_unwind", since = "1.9.0")]
pub fn resume_unwind(payload: Box<Any + Send>) -> ! {
    panicking::update_count_then_panic(payload)
}
