//! Utilities for dynamic typing or type reflection.
//!
//! # `Any` and `TypeId`
//!
//! `Any` itself can be used to get a `TypeId`, and has more features when used
//! as a trait object. As `&dyn Any` (a borrowed trait object), it has the `is`
//! and `downcast_ref` methods, to test if the contained value is of a given type,
//! and to get a reference to the inner value as a type. As `&mut dyn Any`, there
//! is also the `downcast_mut` method, for getting a mutable reference to the
//! inner value. `Box<dyn Any>` adds the `downcast` method, which attempts to
//! convert to a `Box<T>`. See the [`Box`] documentation for the full details.
//!
//! Note that `&dyn Any` is limited to testing whether a value is of a specified
//! concrete type, and cannot be used to test whether a type implements a trait.
//!
//! [`Box`]: ../../std/boxed/struct.Box.html
//!
//! # Smart pointers and `dyn Any`
//!
//! One piece of behavior to keep in mind when using `Any` as a trait object,
//! especially with types like `Box<dyn Any>` or `Arc<dyn Any>`, is that simply
//! calling `.type_id()` on the value will produce the `TypeId` of the
//! *container*, not the underlying trait object. This can be avoided by
//! converting the smart pointer into a `&dyn Any` instead, which will return
//! the object's `TypeId`. For example:
//!
//! ```
//! use std::any::{Any, TypeId};
//!
//! let boxed: Box<dyn Any> = Box::new(3_i32);
//!
//! // You're more likely to want this:
//! let actual_id = (&*boxed).type_id();
//! // ... than this:
//! let boxed_id = boxed.type_id();
//!
//! assert_eq!(actual_id, TypeId::of::<i32>());
//! assert_eq!(boxed_id, TypeId::of::<Box<dyn Any>>());
//! ```
//!
//! ## Examples
//!
//! Consider a situation where we want to log out a value passed to a function.
//! We know the value we're working on implements Debug, but we don't know its
//! concrete type. We want to give special treatment to certain types: in this
//! case printing out the length of String values prior to their value.
//! We don't know the concrete type of our value at compile time, so we need to
//! use runtime reflection instead.
//!
//! ```rust
//! use std::fmt::Debug;
//! use std::any::Any;
//!
//! // Logger function for any type that implements Debug.
//! fn log<T: Any + Debug>(value: &T) {
//!     let value_any = value as &dyn Any;
//!
//!     // Try to convert our value to a `String`. If successful, we want to
//!     // output the `String`'s length as well as its value. If not, it's a
//!     // different type: just print it out unadorned.
//!     match value_any.downcast_ref::<String>() {
//!         Some(as_string) => {
//!             println!("String ({}): {}", as_string.len(), as_string);
//!         }
//!         None => {
//!             println!("{value:?}");
//!         }
//!     }
//! }
//!
//! // This function wants to log its parameter out prior to doing work with it.
//! fn do_work<T: Any + Debug>(value: &T) {
//!     log(value);
//!     // ...do some other work
//! }
//!
//! fn main() {
//!     let my_string = "Hello World".to_string();
//!     do_work(&my_string);
//!
//!     let my_i8: i8 = 100;
//!     do_work(&my_i8);
//! }
//! ```
//!
//! # `Provider` and `Demand`
//!
//! `Provider` and the associated APIs support generic, type-driven access to data, and a mechanism
//! for implementers to provide such data. The key parts of the interface are the `Provider`
//! trait for objects which can provide data, and the [`request_value`] and [`request_ref`]
//! functions for requesting data from an object which implements `Provider`. Generally, end users
//! should not call `request_*` directly, they are helper functions for intermediate implementers
//! to use to implement a user-facing interface. This is purely for the sake of ergonomics, there is
//! no safety concern here; intermediate implementers can typically support methods rather than
//! free functions and use more specific names.
//!
//! Typically, a data provider is a trait object of a trait which extends `Provider`. A user will
//! request data from a trait object by specifying the type of the data.
//!
//! ## Data flow
//!
//! * A user requests an object of a specific type, which is delegated to `request_value` or
//!   `request_ref`
//! * `request_*` creates a `Demand` object and passes it to `Provider::provide`
//! * The data provider's implementation of `Provider::provide` tries providing values of
//!   different types using `Demand::provide_*`. If the type matches the type requested by
//!   the user, the value will be stored in the `Demand` object.
//! * `request_*` unpacks the `Demand` object and returns any stored value to the user.
//!
//! ## Examples
//!
//! ```
//! # #![feature(provide_any)]
//! use std::any::{Provider, Demand, request_ref};
//!
//! // Definition of MyTrait, a data provider.
//! trait MyTrait: Provider {
//!     // ...
//! }
//!
//! // Methods on `MyTrait` trait objects.
//! impl dyn MyTrait + '_ {
//!     /// Get a reference to a field of the implementing struct.
//!     pub fn get_context_by_ref<T: ?Sized + 'static>(&self) -> Option<&T> {
//!         request_ref::<T>(self)
//!     }
//! }
//!
//! // Downstream implementation of `MyTrait` and `Provider`.
//! # struct SomeConcreteType { some_string: String }
//! impl MyTrait for SomeConcreteType {
//!     // ...
//! }
//!
//! impl Provider for SomeConcreteType {
//!     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
//!         // Provide a string reference. We could provide multiple values with
//!         // different types here.
//!         demand.provide_ref::<String>(&self.some_string);
//!     }
//! }
//!
//! // Downstream usage of `MyTrait`.
//! fn use_my_trait(obj: &dyn MyTrait) {
//!     // Request a &String from obj.
//!     let _ = obj.get_context_by_ref::<String>().unwrap();
//! }
//! ```
//!
//! In this example, if the concrete type of `obj` in `use_my_trait` is `SomeConcreteType`, then
//! the `get_context_by_ref` call will return a reference to `obj.some_string` with type `&String`.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fmt;
use crate::hash;
use crate::intrinsics;

///////////////////////////////////////////////////////////////////////////////
// Any trait
///////////////////////////////////////////////////////////////////////////////

/// A trait to emulate dynamic typing.
///
/// Most types implement `Any`. However, any type which contains a non-`'static` reference does not.
/// See the [module-level documentation][mod] for more details.
///
/// [mod]: crate::any
// This trait is not unsafe, though we rely on the specifics of it's sole impl's
// `type_id` function in unsafe code (e.g., `downcast`). Normally, that would be
// a problem, but because the only impl of `Any` is a blanket implementation, no
// other code can implement `Any`.
//
// We could plausibly make this trait unsafe -- it would not cause breakage,
// since we control all the implementations -- but we choose not to as that's
// both not really necessary and may confuse users about the distinction of
// unsafe traits and unsafe methods (i.e., `type_id` would still be safe to call,
// but we would likely want to indicate as such in documentation).
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Any")]
pub trait Any: 'static {
    /// Gets the `TypeId` of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::{Any, TypeId};
    ///
    /// fn is_string(s: &dyn Any) -> bool {
    ///     TypeId::of::<String>() == s.type_id()
    /// }
    ///
    /// assert_eq!(is_string(&0), false);
    /// assert_eq!(is_string(&"cookie monster".to_string()), true);
    /// ```
    #[stable(feature = "get_type_id", since = "1.34.0")]
    fn type_id(&self) -> TypeId;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: 'static + ?Sized> Any for T {
    fn type_id(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

///////////////////////////////////////////////////////////////////////////////
// Extension methods for Any trait objects.
///////////////////////////////////////////////////////////////////////////////

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for dyn Any {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}

// Ensure that the result of e.g., joining a thread can be printed and
// hence used with `unwrap`. May eventually no longer be needed if
// dispatch works with upcasting.
#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for dyn Any + Send {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}

#[stable(feature = "any_send_sync_methods", since = "1.28.0")]
impl fmt::Debug for dyn Any + Send + Sync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}

impl dyn Any {
    /// Returns `true` if the inner type is the same as `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn is_string(s: &dyn Any) {
    ///     if s.is::<String>() {
    ///         println!("It's a string!");
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// is_string(&0);
    /// is_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is<T: Any>(&self) -> bool {
        // Get `TypeId` of the type this function is instantiated with.
        let t = TypeId::of::<T>();

        // Get `TypeId` of the type in the trait object (`self`).
        let concrete = self.type_id();

        // Compare both `TypeId`s on equality.
        t == concrete
    }

    /// Returns some reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(s: &dyn Any) {
    ///     if let Some(string) = s.downcast_ref::<String>() {
    ///         println!("It's a string({}): '{}'", string.len(), string);
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// print_if_string(&0);
    /// print_if_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        if self.is::<T>() {
            // SAFETY: just checked whether we are pointing to the correct type, and we can rely on
            // that check for memory safety because we have implemented Any for all types; no other
            // impls can exist as they would conflict with our impl.
            unsafe { Some(self.downcast_ref_unchecked()) }
        } else {
            None
        }
    }

    /// Returns some mutable reference to the inner value if it is of type `T`, or
    /// `None` if it isn't.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn modify_if_u32(s: &mut dyn Any) {
    ///     if let Some(num) = s.downcast_mut::<u32>() {
    ///         *num = 42;
    ///     }
    /// }
    ///
    /// let mut x = 10u32;
    /// let mut s = "starlord".to_string();
    ///
    /// modify_if_u32(&mut x);
    /// modify_if_u32(&mut s);
    ///
    /// assert_eq!(x, 42);
    /// assert_eq!(&s, "starlord");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        if self.is::<T>() {
            // SAFETY: just checked whether we are pointing to the correct type, and we can rely on
            // that check for memory safety because we have implemented Any for all types; no other
            // impls can exist as they would conflict with our impl.
            unsafe { Some(self.downcast_mut_unchecked()) }
        } else {
            None
        }
    }

    /// Returns a reference to the inner value as type `dyn T`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     assert_eq!(*x.downcast_ref_unchecked::<usize>(), 1);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// The contained value must be of type `T`. Calling this method
    /// with the incorrect type is *undefined behavior*.
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: Any>(&self) -> &T {
        debug_assert!(self.is::<T>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(self as *const dyn Any as *const T) }
    }

    /// Returns a mutable reference to the inner value as type `dyn T`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let mut x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     *x.downcast_mut_unchecked::<usize>() += 1;
    /// }
    ///
    /// assert_eq!(*x.downcast_ref::<usize>().unwrap(), 2);
    /// ```
    ///
    /// # Safety
    ///
    /// The contained value must be of type `T`. Calling this method
    /// with the incorrect type is *undefined behavior*.
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_mut_unchecked<T: Any>(&mut self) -> &mut T {
        debug_assert!(self.is::<T>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &mut *(self as *mut dyn Any as *mut T) }
    }
}

impl dyn Any + Send {
    /// Forwards to the method defined on the type `dyn Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn is_string(s: &(dyn Any + Send)) {
    ///     if s.is::<String>() {
    ///         println!("It's a string!");
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// is_string(&0);
    /// is_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn is<T: Any>(&self) -> bool {
        <dyn Any>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(s: &(dyn Any + Send)) {
    ///     if let Some(string) = s.downcast_ref::<String>() {
    ///         println!("It's a string({}): '{}'", string.len(), string);
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// print_if_string(&0);
    /// print_if_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        <dyn Any>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn modify_if_u32(s: &mut (dyn Any + Send)) {
    ///     if let Some(num) = s.downcast_mut::<u32>() {
    ///         *num = 42;
    ///     }
    /// }
    ///
    /// let mut x = 10u32;
    /// let mut s = "starlord".to_string();
    ///
    /// modify_if_u32(&mut x);
    /// modify_if_u32(&mut s);
    ///
    /// assert_eq!(x, 42);
    /// assert_eq!(&s, "starlord");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        <dyn Any>::downcast_mut::<T>(self)
    }

    /// Forwards to the method defined on the type `dyn Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     assert_eq!(*x.downcast_ref_unchecked::<usize>(), 1);
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// Same as the method on the type `dyn Any`.
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: Any>(&self) -> &T {
        // SAFETY: guaranteed by caller
        unsafe { <dyn Any>::downcast_ref_unchecked::<T>(self) }
    }

    /// Forwards to the method defined on the type `dyn Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let mut x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     *x.downcast_mut_unchecked::<usize>() += 1;
    /// }
    ///
    /// assert_eq!(*x.downcast_ref::<usize>().unwrap(), 2);
    /// ```
    ///
    /// # Safety
    ///
    /// Same as the method on the type `dyn Any`.
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_mut_unchecked<T: Any>(&mut self) -> &mut T {
        // SAFETY: guaranteed by caller
        unsafe { <dyn Any>::downcast_mut_unchecked::<T>(self) }
    }
}

impl dyn Any + Send + Sync {
    /// Forwards to the method defined on the type `Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn is_string(s: &(dyn Any + Send + Sync)) {
    ///     if s.is::<String>() {
    ///         println!("It's a string!");
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// is_string(&0);
    /// is_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "any_send_sync_methods", since = "1.28.0")]
    #[inline]
    pub fn is<T: Any>(&self) -> bool {
        <dyn Any>::is::<T>(self)
    }

    /// Forwards to the method defined on the type `Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(s: &(dyn Any + Send + Sync)) {
    ///     if let Some(string) = s.downcast_ref::<String>() {
    ///         println!("It's a string({}): '{}'", string.len(), string);
    ///     } else {
    ///         println!("Not a string...");
    ///     }
    /// }
    ///
    /// print_if_string(&0);
    /// print_if_string(&"cookie monster".to_string());
    /// ```
    #[stable(feature = "any_send_sync_methods", since = "1.28.0")]
    #[inline]
    pub fn downcast_ref<T: Any>(&self) -> Option<&T> {
        <dyn Any>::downcast_ref::<T>(self)
    }

    /// Forwards to the method defined on the type `Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn modify_if_u32(s: &mut (dyn Any + Send + Sync)) {
    ///     if let Some(num) = s.downcast_mut::<u32>() {
    ///         *num = 42;
    ///     }
    /// }
    ///
    /// let mut x = 10u32;
    /// let mut s = "starlord".to_string();
    ///
    /// modify_if_u32(&mut x);
    /// modify_if_u32(&mut s);
    ///
    /// assert_eq!(x, 42);
    /// assert_eq!(&s, "starlord");
    /// ```
    #[stable(feature = "any_send_sync_methods", since = "1.28.0")]
    #[inline]
    pub fn downcast_mut<T: Any>(&mut self) -> Option<&mut T> {
        <dyn Any>::downcast_mut::<T>(self)
    }

    /// Forwards to the method defined on the type `Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     assert_eq!(*x.downcast_ref_unchecked::<usize>(), 1);
    /// }
    /// ```
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: Any>(&self) -> &T {
        // SAFETY: guaranteed by caller
        unsafe { <dyn Any>::downcast_ref_unchecked::<T>(self) }
    }

    /// Forwards to the method defined on the type `Any`.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(downcast_unchecked)]
    ///
    /// use std::any::Any;
    ///
    /// let mut x: Box<dyn Any> = Box::new(1_usize);
    ///
    /// unsafe {
    ///     *x.downcast_mut_unchecked::<usize>() += 1;
    /// }
    ///
    /// assert_eq!(*x.downcast_ref::<usize>().unwrap(), 2);
    /// ```
    #[unstable(feature = "downcast_unchecked", issue = "90850")]
    #[inline]
    pub unsafe fn downcast_mut_unchecked<T: Any>(&mut self) -> &mut T {
        // SAFETY: guaranteed by caller
        unsafe { <dyn Any>::downcast_mut_unchecked::<T>(self) }
    }
}

///////////////////////////////////////////////////////////////////////////////
// TypeID and its methods
///////////////////////////////////////////////////////////////////////////////

/// A `TypeId` represents a globally unique identifier for a type.
///
/// Each `TypeId` is an opaque object which does not allow inspection of what's
/// inside but does allow basic operations such as cloning, comparison,
/// printing, and showing.
///
/// A `TypeId` is currently only available for types which ascribe to `'static`,
/// but this limitation may be removed in the future.
///
/// While `TypeId` implements `Hash`, `PartialOrd`, and `Ord`, it is worth
/// noting that the hashes and ordering will vary between Rust releases. Beware
/// of relying on them inside of your code!
#[derive(Clone, Copy, Debug, Eq, PartialOrd, Ord)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TypeId {
    t: u128,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl PartialEq for TypeId {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t
    }
}

impl TypeId {
    /// Returns the `TypeId` of the type this generic function has been
    /// instantiated with.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::{Any, TypeId};
    ///
    /// fn is_string<T: ?Sized + Any>(_s: &T) -> bool {
    ///     TypeId::of::<String>() == TypeId::of::<T>()
    /// }
    ///
    /// assert_eq!(is_string(&0), false);
    /// assert_eq!(is_string(&"cookie monster".to_string()), true);
    /// ```
    #[must_use]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_type_id", issue = "77125")]
    pub const fn of<T: ?Sized + 'static>() -> TypeId {
        #[cfg(bootstrap)]
        let t = intrinsics::type_id::<T>() as u128;
        #[cfg(not(bootstrap))]
        let t: u128 = intrinsics::type_id::<T>();
        TypeId { t }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl hash::Hash for TypeId {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        // We only hash the lower 64 bits of our (128 bit) internal numeric ID,
        // because:
        // - The hashing algorithm which backs `TypeId` is expected to be
        //   unbiased and high quality, meaning further mixing would be somewhat
        //   redundant compared to choosing (the lower) 64 bits arbitrarily.
        // - `Hasher::finish` returns a u64 anyway, so the extra entropy we'd
        //   get from hashing the full value would probably not be useful
        //   (especially given the previous point about the lower 64 bits being
        //   high quality on their own).
        // - It is correct to do so -- only hashing a subset of `self` is still
        //   with an `Eq` implementation that considers the entire value, as
        //   ours does.
        (self.t as u64).hash(state);
    }
}

/// Returns the name of a type as a string slice.
///
/// # Note
///
/// This is intended for diagnostic use. The exact contents and format of the
/// string returned are not specified, other than being a best-effort
/// description of the type. For example, amongst the strings
/// that `type_name::<Option<String>>()` might return are `"Option<String>"` and
/// `"std::option::Option<std::string::String>"`.
///
/// The returned string must not be considered to be a unique identifier of a
/// type as multiple types may map to the same type name. Similarly, there is no
/// guarantee that all parts of a type will appear in the returned string: for
/// example, lifetime specifiers are currently not included. In addition, the
/// output may change between versions of the compiler.
///
/// The current implementation uses the same infrastructure as compiler
/// diagnostics and debuginfo, but this is not guaranteed.
///
/// # Examples
///
/// ```rust
/// assert_eq!(
///     std::any::type_name::<Option<String>>(),
///     "core::option::Option<alloc::string::String>",
/// );
/// ```
#[must_use]
#[stable(feature = "type_name", since = "1.38.0")]
#[rustc_const_unstable(feature = "const_type_name", issue = "63084")]
pub const fn type_name<T: ?Sized>() -> &'static str {
    intrinsics::type_name::<T>()
}

/// Returns the name of the type of the pointed-to value as a string slice.
/// This is the same as `type_name::<T>()`, but can be used where the type of a
/// variable is not easily available.
///
/// # Note
///
/// This is intended for diagnostic use. The exact contents and format of the
/// string are not specified, other than being a best-effort description of the
/// type. For example, `type_name_of_val::<Option<String>>(None)` could return
/// `"Option<String>"` or `"std::option::Option<std::string::String>"`, but not
/// `"foobar"`. In addition, the output may change between versions of the
/// compiler.
///
/// This function does not resolve trait objects,
/// meaning that `type_name_of_val(&7u32 as &dyn Debug)`
/// may return `"dyn Debug"`, but not `"u32"`.
///
/// The type name should not be considered a unique identifier of a type;
/// multiple types may share the same type name.
///
/// The current implementation uses the same infrastructure as compiler
/// diagnostics and debuginfo, but this is not guaranteed.
///
/// # Examples
///
/// Prints the default integer and float types.
///
/// ```rust
/// #![feature(type_name_of_val)]
/// use std::any::type_name_of_val;
///
/// let x = 1;
/// println!("{}", type_name_of_val(&x));
/// let y = 1.0;
/// println!("{}", type_name_of_val(&y));
/// ```
#[must_use]
#[unstable(feature = "type_name_of_val", issue = "66359")]
#[rustc_const_unstable(feature = "const_type_name", issue = "63084")]
pub const fn type_name_of_val<T: ?Sized>(_val: &T) -> &'static str {
    type_name::<T>()
}

///////////////////////////////////////////////////////////////////////////////
// Provider trait
///////////////////////////////////////////////////////////////////////////////

/// Trait implemented by a type which can dynamically provide values based on type.
#[unstable(feature = "provide_any", issue = "96024")]
pub trait Provider {
    /// Data providers should implement this method to provide *all* values they are able to
    /// provide by using `demand`.
    ///
    /// Note that the `provide_*` methods on `Demand` have short-circuit semantics: if an earlier
    /// method has successfully provided a value, then later methods will not get an opportunity to
    /// provide.
    ///
    /// # Examples
    ///
    /// Provides a reference to a field with type `String` as a `&str`, and a value of
    /// type `i32`.
    ///
    /// ```rust
    /// # #![feature(provide_any)]
    /// use std::any::{Provider, Demand};
    /// # struct SomeConcreteType { field: String, num_field: i32 }
    ///
    /// impl Provider for SomeConcreteType {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         demand.provide_ref::<str>(&self.field)
    ///             .provide_value::<i32>(self.num_field);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    fn provide<'a>(&'a self, demand: &mut Demand<'a>);
}

/// Request a value from the `Provider`.
///
/// # Examples
///
/// Get a string value from a provider.
///
/// ```rust
/// # #![feature(provide_any)]
/// use std::any::{Provider, request_value};
///
/// fn get_string(provider: &impl Provider) -> String {
///     request_value::<String>(provider).unwrap()
/// }
/// ```
#[unstable(feature = "provide_any", issue = "96024")]
pub fn request_value<'a, T>(provider: &'a (impl Provider + ?Sized)) -> Option<T>
where
    T: 'static,
{
    request_by_type_tag::<'a, tags::Value<T>>(provider)
}

/// Request a reference from the `Provider`.
///
/// # Examples
///
/// Get a string reference from a provider.
///
/// ```rust
/// # #![feature(provide_any)]
/// use std::any::{Provider, request_ref};
///
/// fn get_str(provider: &impl Provider) -> &str {
///     request_ref::<str>(provider).unwrap()
/// }
/// ```
#[unstable(feature = "provide_any", issue = "96024")]
pub fn request_ref<'a, T>(provider: &'a (impl Provider + ?Sized)) -> Option<&'a T>
where
    T: 'static + ?Sized,
{
    request_by_type_tag::<'a, tags::Ref<tags::MaybeSizedValue<T>>>(provider)
}

/// Request a specific value by tag from the `Provider`.
fn request_by_type_tag<'a, I>(provider: &'a (impl Provider + ?Sized)) -> Option<I::Reified>
where
    I: tags::Type<'a>,
{
    let mut tagged = TaggedOption::<'a, I>(None);
    provider.provide(tagged.as_demand());
    tagged.0
}

///////////////////////////////////////////////////////////////////////////////
// Demand and its methods
///////////////////////////////////////////////////////////////////////////////

/// A helper object for providing data by type.
///
/// A data provider provides values by calling this type's provide methods.
#[unstable(feature = "provide_any", issue = "96024")]
#[cfg_attr(not(doc), repr(transparent))] // work around https://github.com/rust-lang/rust/issues/90435
pub struct Demand<'a>(dyn Erased<'a> + 'a);

impl<'a> Demand<'a> {
    /// Create a new `&mut Demand` from a `&mut dyn Erased` trait object.
    fn new<'b>(erased: &'b mut (dyn Erased<'a> + 'a)) -> &'b mut Demand<'a> {
        // SAFETY: transmuting `&mut (dyn Erased<'a> + 'a)` to `&mut Demand<'a>` is safe since
        // `Demand` is repr(transparent).
        unsafe { &mut *(erased as *mut dyn Erased<'a> as *mut Demand<'a>) }
    }

    /// Provide a value or other type with only static lifetimes.
    ///
    /// # Examples
    ///
    /// Provides an `u8`.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    /// # struct SomeConcreteType { field: u8 }
    ///
    /// impl Provider for SomeConcreteType {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         demand.provide_value::<u8>(self.field);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn provide_value<T>(&mut self, value: T) -> &mut Self
    where
        T: 'static,
    {
        self.provide::<tags::Value<T>>(value)
    }

    /// Provide a value or other type with only static lifetimes computed using a closure.
    ///
    /// # Examples
    ///
    /// Provides a `String` by cloning.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    /// # struct SomeConcreteType { field: String }
    ///
    /// impl Provider for SomeConcreteType {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         demand.provide_value_with::<String>(|| self.field.clone());
    ///     }
    /// }
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn provide_value_with<T>(&mut self, fulfil: impl FnOnce() -> T) -> &mut Self
    where
        T: 'static,
    {
        self.provide_with::<tags::Value<T>>(fulfil)
    }

    /// Provide a reference. The referee type must be bounded by `'static`,
    /// but may be unsized.
    ///
    /// # Examples
    ///
    /// Provides a reference to a field as a `&str`.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    /// # struct SomeConcreteType { field: String }
    ///
    /// impl Provider for SomeConcreteType {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         demand.provide_ref::<str>(&self.field);
    ///     }
    /// }
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn provide_ref<T: ?Sized + 'static>(&mut self, value: &'a T) -> &mut Self {
        self.provide::<tags::Ref<tags::MaybeSizedValue<T>>>(value)
    }

    /// Provide a reference computed using a closure. The referee type
    /// must be bounded by `'static`, but may be unsized.
    ///
    /// # Examples
    ///
    /// Provides a reference to a field as a `&str`.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    /// # struct SomeConcreteType { business: String, party: String }
    /// # fn today_is_a_weekday() -> bool { true }
    ///
    /// impl Provider for SomeConcreteType {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         demand.provide_ref_with::<str>(|| {
    ///             if today_is_a_weekday() {
    ///                 &self.business
    ///             } else {
    ///                 &self.party
    ///             }
    ///         });
    ///     }
    /// }
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn provide_ref_with<T: ?Sized + 'static>(
        &mut self,
        fulfil: impl FnOnce() -> &'a T,
    ) -> &mut Self {
        self.provide_with::<tags::Ref<tags::MaybeSizedValue<T>>>(fulfil)
    }

    /// Provide a value with the given `Type` tag.
    fn provide<I>(&mut self, value: I::Reified) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        if let Some(res @ TaggedOption(None)) = self.0.downcast_mut::<I>() {
            res.0 = Some(value);
        }
        self
    }

    /// Provide a value with the given `Type` tag, using a closure to prevent unnecessary work.
    fn provide_with<I>(&mut self, fulfil: impl FnOnce() -> I::Reified) -> &mut Self
    where
        I: tags::Type<'a>,
    {
        if let Some(res @ TaggedOption(None)) = self.0.downcast_mut::<I>() {
            res.0 = Some(fulfil());
        }
        self
    }

    /// Check if the `Demand` would be satisfied if provided with a
    /// value of the specified type. If the type does not match or has
    /// already been provided, returns false.
    ///
    /// # Examples
    ///
    /// Check if an `u8` still needs to be provided and then provides
    /// it.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    ///
    /// struct Parent(Option<u8>);
    ///
    /// impl Provider for Parent {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         if let Some(v) = self.0 {
    ///             demand.provide_value::<u8>(v);
    ///         }
    ///     }
    /// }
    ///
    /// struct Child {
    ///     parent: Parent,
    /// }
    ///
    /// impl Child {
    ///     // Pretend that this takes a lot of resources to evaluate.
    ///     fn an_expensive_computation(&self) -> Option<u8> {
    ///         Some(99)
    ///     }
    /// }
    ///
    /// impl Provider for Child {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         // In general, we don't know if this call will provide
    ///         // an `u8` value or not...
    ///         self.parent.provide(demand);
    ///
    ///         // ...so we check to see if the `u8` is needed before
    ///         // we run our expensive computation.
    ///         if demand.would_be_satisfied_by_value_of::<u8>() {
    ///             if let Some(v) = self.an_expensive_computation() {
    ///                 demand.provide_value::<u8>(v);
    ///             }
    ///         }
    ///
    ///         // The demand will be satisfied now, regardless of if
    ///         // the parent provided the value or we did.
    ///         assert!(!demand.would_be_satisfied_by_value_of::<u8>());
    ///     }
    /// }
    ///
    /// let parent = Parent(Some(42));
    /// let child = Child { parent };
    /// assert_eq!(Some(42), std::any::request_value::<u8>(&child));
    ///
    /// let parent = Parent(None);
    /// let child = Child { parent };
    /// assert_eq!(Some(99), std::any::request_value::<u8>(&child));
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn would_be_satisfied_by_value_of<T>(&self) -> bool
    where
        T: 'static,
    {
        self.would_be_satisfied_by::<tags::Value<T>>()
    }

    /// Check if the `Demand` would be satisfied if provided with a
    /// reference to a value of the specified type. If the type does
    /// not match or has already been provided, returns false.
    ///
    /// # Examples
    ///
    /// Check if a `&str` still needs to be provided and then provides
    /// it.
    ///
    /// ```rust
    /// #![feature(provide_any)]
    ///
    /// use std::any::{Provider, Demand};
    ///
    /// struct Parent(Option<String>);
    ///
    /// impl Provider for Parent {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         if let Some(v) = &self.0 {
    ///             demand.provide_ref::<str>(v);
    ///         }
    ///     }
    /// }
    ///
    /// struct Child {
    ///     parent: Parent,
    ///     name: String,
    /// }
    ///
    /// impl Child {
    ///     // Pretend that this takes a lot of resources to evaluate.
    ///     fn an_expensive_computation(&self) -> Option<&str> {
    ///         Some(&self.name)
    ///     }
    /// }
    ///
    /// impl Provider for Child {
    ///     fn provide<'a>(&'a self, demand: &mut Demand<'a>) {
    ///         // In general, we don't know if this call will provide
    ///         // a `str` reference or not...
    ///         self.parent.provide(demand);
    ///
    ///         // ...so we check to see if the `&str` is needed before
    ///         // we run our expensive computation.
    ///         if demand.would_be_satisfied_by_ref_of::<str>() {
    ///             if let Some(v) = self.an_expensive_computation() {
    ///                 demand.provide_ref::<str>(v);
    ///             }
    ///         }
    ///
    ///         // The demand will be satisfied now, regardless of if
    ///         // the parent provided the reference or we did.
    ///         assert!(!demand.would_be_satisfied_by_ref_of::<str>());
    ///     }
    /// }
    ///
    /// let parent = Parent(Some("parent".into()));
    /// let child = Child { parent, name: "child".into() };
    /// assert_eq!(Some("parent"), std::any::request_ref::<str>(&child));
    ///
    /// let parent = Parent(None);
    /// let child = Child { parent, name: "child".into() };
    /// assert_eq!(Some("child"), std::any::request_ref::<str>(&child));
    /// ```
    #[unstable(feature = "provide_any", issue = "96024")]
    pub fn would_be_satisfied_by_ref_of<T>(&self) -> bool
    where
        T: ?Sized + 'static,
    {
        self.would_be_satisfied_by::<tags::Ref<tags::MaybeSizedValue<T>>>()
    }

    fn would_be_satisfied_by<I>(&self) -> bool
    where
        I: tags::Type<'a>,
    {
        matches!(self.0.downcast::<I>(), Some(TaggedOption(None)))
    }
}

#[unstable(feature = "provide_any", issue = "96024")]
impl<'a> fmt::Debug for Demand<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Demand").finish_non_exhaustive()
    }
}

///////////////////////////////////////////////////////////////////////////////
// Type tags
///////////////////////////////////////////////////////////////////////////////

mod tags {
    //! Type tags are used to identify a type using a separate value. This module includes type tags
    //! for some very common types.
    //!
    //! Currently type tags are not exposed to the user. But in the future, if you want to use the
    //! Provider API with more complex types (typically those including lifetime parameters), you
    //! will need to write your own tags.

    use crate::marker::PhantomData;

    /// This trait is implemented by specific tag types in order to allow
    /// describing a type which can be requested for a given lifetime `'a`.
    ///
    /// A few example implementations for type-driven tags can be found in this
    /// module, although crates may also implement their own tags for more
    /// complex types with internal lifetimes.
    pub trait Type<'a>: Sized + 'static {
        /// The type of values which may be tagged by this tag for the given
        /// lifetime.
        type Reified: 'a;
    }

    /// Similar to the [`Type`] trait, but represents a type which may be unsized (i.e., has a
    /// `?Sized` bound). E.g., `str`.
    pub trait MaybeSizedType<'a>: Sized + 'static {
        type Reified: 'a + ?Sized;
    }

    impl<'a, T: Type<'a>> MaybeSizedType<'a> for T {
        type Reified = T::Reified;
    }

    /// Type-based tag for types bounded by `'static`, i.e., with no borrowed elements.
    #[derive(Debug)]
    pub struct Value<T: 'static>(PhantomData<T>);

    impl<'a, T: 'static> Type<'a> for Value<T> {
        type Reified = T;
    }

    /// Type-based tag similar to [`Value`] but which may be unsized (i.e., has a `?Sized` bound).
    #[derive(Debug)]
    pub struct MaybeSizedValue<T: ?Sized + 'static>(PhantomData<T>);

    impl<'a, T: ?Sized + 'static> MaybeSizedType<'a> for MaybeSizedValue<T> {
        type Reified = T;
    }

    /// Type-based tag for reference types (`&'a T`, where T is represented by
    /// `<I as MaybeSizedType<'a>>::Reified`.
    #[derive(Debug)]
    pub struct Ref<I>(PhantomData<I>);

    impl<'a, I: MaybeSizedType<'a>> Type<'a> for Ref<I> {
        type Reified = &'a I::Reified;
    }
}

/// An `Option` with a type tag `I`.
///
/// Since this struct implements `Erased`, the type can be erased to make a dynamically typed
/// option. The type can be checked dynamically using `Erased::tag_id` and since this is statically
/// checked for the concrete type, there is some degree of type safety.
#[repr(transparent)]
struct TaggedOption<'a, I: tags::Type<'a>>(Option<I::Reified>);

impl<'a, I: tags::Type<'a>> TaggedOption<'a, I> {
    fn as_demand(&mut self) -> &mut Demand<'a> {
        Demand::new(self as &mut (dyn Erased<'a> + 'a))
    }
}

/// Represents a type-erased but identifiable object.
///
/// This trait is exclusively implemented by the `TaggedOption` type.
unsafe trait Erased<'a>: 'a {
    /// The `TypeId` of the erased type.
    fn tag_id(&self) -> TypeId;
}

unsafe impl<'a, I: tags::Type<'a>> Erased<'a> for TaggedOption<'a, I> {
    fn tag_id(&self) -> TypeId {
        TypeId::of::<I>()
    }
}

#[unstable(feature = "provide_any", issue = "96024")]
impl<'a> dyn Erased<'a> + 'a {
    /// Returns some reference to the dynamic value if it is tagged with `I`,
    /// or `None` otherwise.
    #[inline]
    fn downcast<I>(&self) -> Option<&TaggedOption<'a, I>>
    where
        I: tags::Type<'a>,
    {
        if self.tag_id() == TypeId::of::<I>() {
            // SAFETY: Just checked whether we're pointing to an I.
            Some(unsafe { &*(self as *const Self).cast::<TaggedOption<'a, I>>() })
        } else {
            None
        }
    }

    /// Returns some mutable reference to the dynamic value if it is tagged with `I`,
    /// or `None` otherwise.
    #[inline]
    fn downcast_mut<I>(&mut self) -> Option<&mut TaggedOption<'a, I>>
    where
        I: tags::Type<'a>,
    {
        if self.tag_id() == TypeId::of::<I>() {
            // SAFETY: Just checked whether we're pointing to an I.
            Some(unsafe { &mut *(self as *mut Self).cast::<TaggedOption<'a, I>>() })
        } else {
            None
        }
    }
}
