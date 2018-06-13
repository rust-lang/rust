// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pointer type for heap allocation.
//!
//! `Box<T>`, casually referred to as a 'box', provides the simplest form of
//! heap allocation in Rust. Boxes provide ownership for this allocation, and
//! drop their contents when they go out of scope.
//!
//! # Examples
//!
//! Creating a box:
//!
//! ```
//! let x = Box::new(5);
//! ```
//!
//! Creating a recursive data structure:
//!
//! ```
//! #[derive(Debug)]
//! enum List<T> {
//!     Cons(T, Box<List<T>>),
//!     Nil,
//! }
//!
//! fn main() {
//!     let list: List<i32> = List::Cons(1, Box::new(List::Cons(2, Box::new(List::Nil))));
//!     println!("{:?}", list);
//! }
//! ```
//!
//! This will print `Cons(1, Cons(2, Nil))`.
//!
//! Recursive structures must be boxed, because if the definition of `Cons`
//! looked like this:
//!
//! ```compile_fail,E0072
//! # enum List<T> {
//! Cons(T, List<T>),
//! # }
//! ```
//!
//! It wouldn't work. This is because the size of a `List` depends on how many
//! elements are in the list, and so we don't know how much memory to allocate
//! for a `Cons`. By introducing a `Box`, which has a defined size, we know how
//! big `Cons` needs to be.

#![stable(feature = "rust1", since = "1.0.0")]

use core::any::Any;
use core::borrow;
use core::cmp::Ordering;
use core::fmt;
use core::future::Future;
use core::hash::{Hash, Hasher};
use core::iter::FusedIterator;
use core::marker::{Unpin, Unsize};
use core::mem::{self, PinMut};
use core::ops::{CoerceUnsized, Deref, DerefMut, Generator, GeneratorState};
use core::ptr::{self, NonNull, Unique};
use core::task::{Context, Poll, UnsafeTask, TaskObj};
use core::convert::From;

use raw_vec::RawVec;
use str::from_boxed_utf8_unchecked;

/// A pointer type for heap allocation.
///
/// See the [module-level documentation](../../std/boxed/index.html) for more.
#[lang = "owned_box"]
#[fundamental]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Box<T: ?Sized>(Unique<T>);

impl<T> Box<T> {
    /// Allocates memory on the heap and then places `x` into it.
    ///
    /// This doesn't actually allocate if `T` is zero-sized.
    ///
    /// # Examples
    ///
    /// ```
    /// let five = Box::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn new(x: T) -> Box<T> {
        box x
    }
}

impl<T: ?Sized> Box<T> {
    /// Constructs a box from a raw pointer.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `Box`. Specifically, the `Box` destructor will call
    /// the destructor of `T` and free the allocated memory. Since the
    /// way `Box` allocates and releases memory is unspecified, the
    /// only valid pointer to pass to this function is the one taken
    /// from another `Box` via the [`Box::into_raw`] function.
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    ///
    /// [`Box::into_raw`]: struct.Box.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let ptr = Box::into_raw(x);
    /// let x = unsafe { Box::from_raw(ptr) };
    /// ```
    #[stable(feature = "box_raw", since = "1.4.0")]
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        Box(Unique::new_unchecked(raw))
    }

    /// Consumes the `Box`, returning the wrapped raw pointer.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to convert the raw pointer back into a
    /// `Box` with the [`Box::from_raw`] function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_raw(b)` instead of `b.into_raw()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let ptr = Box::into_raw(x);
    /// ```
    #[stable(feature = "box_raw", since = "1.4.0")]
    #[inline]
    pub fn into_raw(b: Box<T>) -> *mut T {
        Box::into_raw_non_null(b).as_ptr()
    }

    /// Consumes the `Box`, returning the wrapped pointer as `NonNull<T>`.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `Box`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to convert the `NonNull<T>` pointer
    /// into a raw pointer and back into a `Box` with the [`Box::from_raw`]
    /// function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::into_raw_non_null(b)`
    /// instead of `b.into_raw_non_null()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(box_into_raw_non_null)]
    ///
    /// fn main() {
    ///     let x = Box::new(5);
    ///     let ptr = Box::into_raw_non_null(x);
    /// }
    /// ```
    #[unstable(feature = "box_into_raw_non_null", issue = "47336")]
    #[inline]
    pub fn into_raw_non_null(b: Box<T>) -> NonNull<T> {
        Box::into_unique(b).into()
    }

    #[unstable(feature = "ptr_internals", issue = "0", reason = "use into_raw_non_null instead")]
    #[inline]
    #[doc(hidden)]
    pub fn into_unique(b: Box<T>) -> Unique<T> {
        let unique = b.0;
        mem::forget(b);
        unique
    }

    /// Consumes and leaks the `Box`, returning a mutable reference,
    /// `&'a mut T`. Here, the lifetime `'a` may be chosen to be `'static`.
    ///
    /// This function is mainly useful for data that lives for the remainder of
    /// the program's life. Dropping the returned reference will cause a memory
    /// leak. If this is not acceptable, the reference should first be wrapped
    /// with the [`Box::from_raw`] function producing a `Box`. This `Box` can
    /// then be dropped which will properly destroy `T` and release the
    /// allocated memory.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `Box::leak(b)` instead of `b.leak()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`Box::from_raw`]: struct.Box.html#method.from_raw
    ///
    /// # Examples
    ///
    /// Simple usage:
    ///
    /// ```
    /// fn main() {
    ///     let x = Box::new(41);
    ///     let static_ref: &'static mut usize = Box::leak(x);
    ///     *static_ref += 1;
    ///     assert_eq!(*static_ref, 42);
    /// }
    /// ```
    ///
    /// Unsized data:
    ///
    /// ```
    /// fn main() {
    ///     let x = vec![1, 2, 3].into_boxed_slice();
    ///     let static_ref = Box::leak(x);
    ///     static_ref[0] = 4;
    ///     assert_eq!(*static_ref, [4, 2, 3]);
    /// }
    /// ```
    #[stable(feature = "box_leak", since = "1.26.0")]
    #[inline]
    pub fn leak<'a>(b: Box<T>) -> &'a mut T
    where
        T: 'a // Technically not needed, but kept to be explicit.
    {
        unsafe { &mut *Box::into_raw(b) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
unsafe impl<#[may_dangle] T: ?Sized> Drop for Box<T> {
    fn drop(&mut self) {
        // FIXME: Do nothing, drop is currently performed by compiler.
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Box<T> {
    /// Creates a `Box<T>`, with the `Default` value for T.
    fn default() -> Box<T> {
        box Default::default()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Box<[T]> {
    fn default() -> Box<[T]> {
        Box::<[T; 0]>::new([])
    }
}

#[stable(feature = "default_box_extra", since = "1.17.0")]
impl Default for Box<str> {
    fn default() -> Box<str> {
        unsafe { from_boxed_utf8_unchecked(Default::default()) }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Clone> Clone for Box<T> {
    /// Returns a new box with a `clone()` of this box's contents.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let y = x.clone();
    /// ```
    #[rustfmt_skip]
    #[inline]
    fn clone(&self) -> Box<T> {
        box { (**self).clone() }
    }
    /// Copies `source`'s contents into `self` without creating a new allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// let mut y = Box::new(10);
    ///
    /// y.clone_from(&x);
    ///
    /// assert_eq!(*y, 5);
    /// ```
    #[inline]
    fn clone_from(&mut self, source: &Box<T>) {
        (**self).clone_from(&(**source));
    }
}


#[stable(feature = "box_slice_clone", since = "1.3.0")]
impl Clone for Box<str> {
    fn clone(&self) -> Self {
        let len = self.len();
        let buf = RawVec::with_capacity(len);
        unsafe {
            ptr::copy_nonoverlapping(self.as_ptr(), buf.ptr(), len);
            from_boxed_utf8_unchecked(buf.into_box())
        }
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool {
        PartialEq::ne(&**self, &**other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Box<T> {
    #[inline]
    fn partial_cmp(&self, other: &Box<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    #[inline]
    fn lt(&self, other: &Box<T>) -> bool {
        PartialOrd::lt(&**self, &**other)
    }
    #[inline]
    fn le(&self, other: &Box<T>) -> bool {
        PartialOrd::le(&**self, &**other)
    }
    #[inline]
    fn ge(&self, other: &Box<T>) -> bool {
        PartialOrd::ge(&**self, &**other)
    }
    #[inline]
    fn gt(&self, other: &Box<T>) -> bool {
        PartialOrd::gt(&**self, &**other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Ord> Ord for Box<T> {
    #[inline]
    fn cmp(&self, other: &Box<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Eq> Eq for Box<T> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + Hash> Hash for Box<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

#[stable(feature = "indirect_hasher_impl", since = "1.22.0")]
impl<T: ?Sized + Hasher> Hasher for Box<T> {
    fn finish(&self) -> u64 {
        (**self).finish()
    }
    fn write(&mut self, bytes: &[u8]) {
        (**self).write(bytes)
    }
    fn write_u8(&mut self, i: u8) {
        (**self).write_u8(i)
    }
    fn write_u16(&mut self, i: u16) {
        (**self).write_u16(i)
    }
    fn write_u32(&mut self, i: u32) {
        (**self).write_u32(i)
    }
    fn write_u64(&mut self, i: u64) {
        (**self).write_u64(i)
    }
    fn write_u128(&mut self, i: u128) {
        (**self).write_u128(i)
    }
    fn write_usize(&mut self, i: usize) {
        (**self).write_usize(i)
    }
    fn write_i8(&mut self, i: i8) {
        (**self).write_i8(i)
    }
    fn write_i16(&mut self, i: i16) {
        (**self).write_i16(i)
    }
    fn write_i32(&mut self, i: i32) {
        (**self).write_i32(i)
    }
    fn write_i64(&mut self, i: i64) {
        (**self).write_i64(i)
    }
    fn write_i128(&mut self, i: i128) {
        (**self).write_i128(i)
    }
    fn write_isize(&mut self, i: isize) {
        (**self).write_isize(i)
    }
}

#[stable(feature = "from_for_ptrs", since = "1.6.0")]
impl<T> From<T> for Box<T> {
    fn from(t: T) -> Self {
        Box::new(t)
    }
}

#[stable(feature = "box_from_slice", since = "1.17.0")]
impl<'a, T: Copy> From<&'a [T]> for Box<[T]> {
    fn from(slice: &'a [T]) -> Box<[T]> {
        let mut boxed = unsafe { RawVec::with_capacity(slice.len()).into_box() };
        boxed.copy_from_slice(slice);
        boxed
    }
}

#[stable(feature = "box_from_slice", since = "1.17.0")]
impl<'a> From<&'a str> for Box<str> {
    #[inline]
    fn from(s: &'a str) -> Box<str> {
        unsafe { from_boxed_utf8_unchecked(Box::from(s.as_bytes())) }
    }
}

#[stable(feature = "boxed_str_conv", since = "1.19.0")]
impl From<Box<str>> for Box<[u8]> {
    #[inline]
    fn from(s: Box<str>) -> Self {
        unsafe { Box::from_raw(Box::into_raw(s) as *mut [u8]) }
    }
}

impl Box<Any> {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Attempt to downcast the box to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(value: Box<Any>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let my_string = "Hello World".to_string();
    ///     print_if_string(Box::new(my_string));
    ///     print_if_string(Box::new(0i8));
    /// }
    /// ```
    pub fn downcast<T: Any>(self) -> Result<Box<T>, Box<Any>> {
        if self.is::<T>() {
            unsafe {
                let raw: *mut Any = Box::into_raw(self);
                Ok(Box::from_raw(raw as *mut T))
            }
        } else {
            Err(self)
        }
    }
}

impl Box<Any + Send> {
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    /// Attempt to downcast the box to a concrete type.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::any::Any;
    ///
    /// fn print_if_string(value: Box<Any + Send>) {
    ///     if let Ok(string) = value.downcast::<String>() {
    ///         println!("String ({}): {}", string.len(), string);
    ///     }
    /// }
    ///
    /// fn main() {
    ///     let my_string = "Hello World".to_string();
    ///     print_if_string(Box::new(my_string));
    ///     print_if_string(Box::new(0i8));
    /// }
    /// ```
    pub fn downcast<T: Any>(self) -> Result<Box<T>, Box<Any + Send>> {
        <Box<Any>>::downcast(self).map_err(|s| unsafe {
            // reapply the Send marker
            Box::from_raw(Box::into_raw(s) as *mut (Any + Send))
        })
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Display + ?Sized> fmt::Display for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: fmt::Debug + ?Sized> fmt::Debug for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> fmt::Pointer for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // It's not possible to extract the inner Uniq directly from the Box,
        // instead we cast it to a *const which aliases the Unique
        let ptr: *const T = &**self;
        fmt::Pointer::fmt(&ptr, f)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &**self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized> Iterator for Box<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> {
        (**self).next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        (**self).nth(n)
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for Box<I> {
    fn next_back(&mut self) -> Option<I::Item> {
        (**self).next_back()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for Box<I> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl<I: FusedIterator + ?Sized> FusedIterator for Box<I> {}


/// `FnBox` is a version of the `FnOnce` intended for use with boxed
/// closure objects. The idea is that where one would normally store a
/// `Box<FnOnce()>` in a data structure, you should use
/// `Box<FnBox()>`. The two traits behave essentially the same, except
/// that a `FnBox` closure can only be called if it is boxed. (Note
/// that `FnBox` may be deprecated in the future if `Box<FnOnce()>`
/// closures become directly usable.)
///
/// # Examples
///
/// Here is a snippet of code which creates a hashmap full of boxed
/// once closures and then removes them one by one, calling each
/// closure as it is removed. Note that the type of the closures
/// stored in the map is `Box<FnBox() -> i32>` and not `Box<FnOnce()
/// -> i32>`.
///
/// ```
/// #![feature(fnbox)]
///
/// use std::boxed::FnBox;
/// use std::collections::HashMap;
///
/// fn make_map() -> HashMap<i32, Box<FnBox() -> i32>> {
///     let mut map: HashMap<i32, Box<FnBox() -> i32>> = HashMap::new();
///     map.insert(1, Box::new(|| 22));
///     map.insert(2, Box::new(|| 44));
///     map
/// }
///
/// fn main() {
///     let mut map = make_map();
///     for i in &[1, 2] {
///         let f = map.remove(&i).unwrap();
///         assert_eq!(f(), i * 22);
///     }
/// }
/// ```
#[rustc_paren_sugar]
#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
pub trait FnBox<A> {
    type Output;

    fn call_box(self: Box<Self>, args: A) -> Self::Output;
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<A, F> FnBox<A> for F
    where F: FnOnce<A>
{
    type Output = F::Output;

    fn call_box(self: Box<F>, args: A) -> F::Output {
        self.call_once(args)
    }
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<'a, A, R> FnOnce<A> for Box<FnBox<A, Output = R> + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> R {
        self.call_box(args)
    }
}

#[unstable(feature = "fnbox",
           reason = "will be deprecated if and when `Box<FnOnce>` becomes usable", issue = "28796")]
impl<'a, A, R> FnOnce<A> for Box<FnBox<A, Output = R> + Send + 'a> {
    type Output = R;

    extern "rust-call" fn call_once(self, args: A) -> R {
        self.call_box(args)
    }
}

#[unstable(feature = "coerce_unsized", issue = "27732")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<Box<U>> for Box<T> {}

#[stable(feature = "box_slice_clone", since = "1.3.0")]
impl<T: Clone> Clone for Box<[T]> {
    fn clone(&self) -> Self {
        let mut new = BoxBuilder {
            data: RawVec::with_capacity(self.len()),
            len: 0,
        };

        let mut target = new.data.ptr();

        for item in self.iter() {
            unsafe {
                ptr::write(target, item.clone());
                target = target.offset(1);
            };

            new.len += 1;
        }

        return unsafe { new.into_box() };

        // Helper type for responding to panics correctly.
        struct BoxBuilder<T> {
            data: RawVec<T>,
            len: usize,
        }

        impl<T> BoxBuilder<T> {
            unsafe fn into_box(self) -> Box<[T]> {
                let raw = ptr::read(&self.data);
                mem::forget(self);
                raw.into_box()
            }
        }

        impl<T> Drop for BoxBuilder<T> {
            fn drop(&mut self) {
                let mut data = self.data.ptr();
                let max = unsafe { data.offset(self.len as isize) };

                while data != max {
                    unsafe {
                        ptr::read(data);
                        data = data.offset(1);
                    }
                }
            }
        }
    }
}

#[stable(feature = "box_borrow", since = "1.1.0")]
impl<T: ?Sized> borrow::Borrow<T> for Box<T> {
    fn borrow(&self) -> &T {
        &**self
    }
}

#[stable(feature = "box_borrow", since = "1.1.0")]
impl<T: ?Sized> borrow::BorrowMut<T> for Box<T> {
    fn borrow_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized> AsRef<T> for Box<T> {
    fn as_ref(&self) -> &T {
        &**self
    }
}

#[stable(since = "1.5.0", feature = "smart_ptr_as_ref")]
impl<T: ?Sized> AsMut<T> for Box<T> {
    fn as_mut(&mut self) -> &mut T {
        &mut **self
    }
}

#[unstable(feature = "generator_trait", issue = "43122")]
impl<T> Generator for Box<T>
    where T: Generator + ?Sized
{
    type Yield = T::Yield;
    type Return = T::Return;
    unsafe fn resume(&mut self) -> GeneratorState<Self::Yield, Self::Return> {
        (**self).resume()
    }
}

/// A pinned, heap allocated reference.
#[unstable(feature = "pin", issue = "49150")]
#[fundamental]
#[repr(transparent)]
pub struct PinBox<T: ?Sized> {
    inner: Box<T>,
}

#[unstable(feature = "pin", issue = "49150")]
impl<T> PinBox<T> {
    /// Allocate memory on the heap, move the data into it and pin it.
    #[unstable(feature = "pin", issue = "49150")]
    pub fn new(data: T) -> PinBox<T> {
        PinBox { inner: Box::new(data) }
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> PinBox<T> {
    /// Get a pinned reference to the data in this PinBox.
    #[inline]
    pub fn as_pin_mut<'a>(&'a mut self) -> PinMut<'a, T> {
        unsafe { PinMut::new_unchecked(&mut *self.inner) }
    }

    /// Constructs a `PinBox` from a raw pointer.
    ///
    /// After calling this function, the raw pointer is owned by the
    /// resulting `PinBox`. Specifically, the `PinBox` destructor will call
    /// the destructor of `T` and free the allocated memory. Since the
    /// way `PinBox` allocates and releases memory is unspecified, the
    /// only valid pointer to pass to this function is the one taken
    /// from another `PinBox` via the [`PinBox::into_raw`] function.
    ///
    /// This function is unsafe because improper use may lead to
    /// memory problems. For example, a double-free may occur if the
    /// function is called twice on the same raw pointer.
    ///
    /// [`PinBox::into_raw`]: struct.PinBox.html#method.into_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pin)]
    /// use std::boxed::PinBox;
    /// let x = PinBox::new(5);
    /// let ptr = PinBox::into_raw(x);
    /// let x = unsafe { PinBox::from_raw(ptr) };
    /// ```
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        PinBox { inner: Box::from_raw(raw) }
    }

    /// Consumes the `PinBox`, returning the wrapped raw pointer.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the `PinBox`. In particular, the
    /// caller should properly destroy `T` and release the memory. The
    /// proper way to do so is to convert the raw pointer back into a
    /// `PinBox` with the [`PinBox::from_raw`] function.
    ///
    /// Note: this is an associated function, which means that you have
    /// to call it as `PinBox::into_raw(b)` instead of `b.into_raw()`. This
    /// is so that there is no conflict with a method on the inner type.
    ///
    /// [`PinBox::from_raw`]: struct.PinBox.html#method.from_raw
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(pin)]
    /// use std::boxed::PinBox;
    /// let x = PinBox::new(5);
    /// let ptr = PinBox::into_raw(x);
    /// ```
    #[inline]
    pub fn into_raw(b: PinBox<T>) -> *mut T {
        Box::into_raw(b.inner)
    }

    /// Get a mutable reference to the data inside this PinBox.
    ///
    /// This function is unsafe. Users must guarantee that the data is never
    /// moved out of this reference.
    #[inline]
    pub unsafe fn get_mut<'a>(this: &'a mut PinBox<T>) -> &'a mut T {
        &mut *this.inner
    }

    /// Convert this PinBox into an unpinned Box.
    ///
    /// This function is unsafe. Users must guarantee that the data is never
    /// moved out of the box.
    #[inline]
    pub unsafe fn unpin(this: PinBox<T>) -> Box<T> {
        this.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> From<Box<T>> for PinBox<T> {
    fn from(boxed: Box<T>) -> PinBox<T> {
        PinBox { inner: boxed }
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: Unpin + ?Sized> From<PinBox<T>> for Box<T> {
    fn from(pinned: PinBox<T>) -> Box<T> {
        pinned.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> Deref for PinBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &*self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: Unpin + ?Sized> DerefMut for PinBox<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut *self.inner
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: fmt::Display + ?Sized> fmt::Display for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&*self.inner, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: fmt::Debug + ?Sized> fmt::Debug for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&*self.inner, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> fmt::Pointer for PinBox<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // It's not possible to extract the inner Uniq directly from the Box,
        // instead we cast it to a *const which aliases the Unique
        let ptr: *const T = &*self.inner;
        fmt::Pointer::fmt(&ptr, f)
    }
}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<PinBox<U>> for PinBox<T> {}

#[unstable(feature = "pin", issue = "49150")]
impl<T: ?Sized> Unpin for PinBox<T> {}

#[unstable(feature = "futures_api", issue = "50547")]
impl<'a, F: ?Sized + Future + Unpin> Future for Box<F> {
    type Output = F::Output;

    fn poll(mut self: PinMut<Self>, cx: &mut Context) -> Poll<Self::Output> {
        PinMut::new(&mut **self).poll(cx)
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
impl<'a, F: ?Sized + Future> Future for PinBox<F> {
    type Output = F::Output;

    fn poll(mut self: PinMut<Self>, cx: &mut Context) -> Poll<Self::Output> {
        self.as_pin_mut().poll(cx)
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
unsafe impl<F: Future<Output = ()> + Send + 'static> UnsafeTask for PinBox<F> {
    fn into_raw(self) -> *mut () {
        PinBox::into_raw(self) as *mut ()
    }

    unsafe fn poll(task: *mut (), cx: &mut Context) -> Poll<()> {
        let ptr = task as *mut F;
        let pin: PinMut<F> = PinMut::new_unchecked(&mut *ptr);
        pin.poll(cx)
    }

    unsafe fn drop(task: *mut ()) {
        drop(PinBox::from_raw(task as *mut F))
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
impl<F: Future<Output = ()> + Send + 'static> From<PinBox<F>> for TaskObj {
    fn from(boxed: PinBox<F>) -> Self {
        TaskObj::new(boxed)
    }
}

#[unstable(feature = "futures_api", issue = "50547")]
impl<F: Future<Output = ()> + Send + 'static> From<Box<F>> for TaskObj {
    fn from(boxed: Box<F>) -> Self {
        TaskObj::new(PinBox::from(boxed))
    }
}
