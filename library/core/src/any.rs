//! This module implements the `Any` trait, which enables dynamic typing
//! of any `'static` type through runtime reflection.
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
//! # Examples
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
//!     // output the String`'s length as well as its value. If not, it's a
//!     // different type: just print it out unadorned.
//!     match value_any.downcast_ref::<String>() {
//!         Some(as_string) => {
//!             println!("String ({}): {}", as_string.len(), as_string);
//!         }
//!         None => {
//!             println!("{:?}", value);
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

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fmt;
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
    /// For a safe alternative see [`downcast_ref`].
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
    #[inline]
    pub unsafe fn downcast_ref_unchecked<T: Any>(&self) -> &T {
        debug_assert!(self.is::<T>());
        // SAFETY: caller guarantees that T is the correct type
        unsafe { &*(self as *const dyn Any as *const T) }
    }

    /// Returns a mutable reference to the inner value as type `dyn T`.
    ///
    /// For a safe alternative see [`downcast_mut`].
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
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
    #[unstable(feature = "downcast_unchecked", issue = "none")]
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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct TypeId {
    t: u64,
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
        TypeId { t: intrinsics::type_id::<T>() }
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
