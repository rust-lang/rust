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
//! Boxes are useful in two situations: recursive data structures, and
//! occasionally when returning data. [The Pointer chapter of the
//! Book](../../../book/pointers.html#best-practices-1) explains these cases in
//! detail.
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
//! This will print `Cons(1, Box(Cons(2, Box(Nil))))`.

#![stable(feature = "rust1", since = "1.0.0")]

use core::prelude::*;

use heap;

use core::any::Any;
use core::cmp::Ordering;
use core::default::Default;
use core::error::{Error, FromError};
use core::fmt;
use core::hash::{self, Hash};
use core::marker;
use core::mem;
use core::ops::{Deref, DerefMut};
use core::ops::{Placer, Boxed, Place, InPlace, BoxPlace};
use core::ptr::Unique;
use core::raw::TraitObject;

/// A value that represents the heap. This is the default place that the `box`
/// keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
/// ```rust
/// #![feature(box_syntax)]
/// use std::boxed::HEAP;
///
/// fn main() {
///     let foo = box(HEAP) 5;
///     let foo = box 5;
/// }
/// ```
#[lang = "exchange_heap"]
#[unstable(feature = "alloc",
           reason = "may be renamed; uncertain about custom allocator design")]
pub const HEAP: ExchangeHeapSingleton =
    ExchangeHeapSingleton { _force_singleton: () };

/// This the singleton type used solely for `boxed::HEAP`.
pub struct ExchangeHeapSingleton { _force_singleton: () }
impl Copy for ExchangeHeapSingleton { }

/// A pointer type for heap allocation.
///
/// See the [module-level documentation](../../std/boxed/index.html) for more.
#[lang = "owned_box"]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Box<T: ?Sized>(Unique<T>);

/// `IntermediateBox` represents uninitialized backing storage for `Box`.
///
/// FIXME (pnkfelix): Ideally we would just reuse `Box<T>` instead of
/// introducing a separate `IntermediateBox<T>`; but then you hit
/// issues when you e.g. attempt to destructure an instance of `Box`,
/// since it is a lang item and so it gets special handling by the
/// compiler.  Easier just to make this parallel type for now.
///
/// FIXME (pnkfelix): Currently the `box` protocol only supports
/// creating instances of sized types. This IntermediateBox is
/// designed to be forward-compatible with a future protocol that
/// supports creating instances of unsized types; that is why the type
/// parameter has the `?Sized` generalization marker, and is also why
/// this carries an explicit size. However, it probably does not need
/// to carry the explicit alignment; that is just a work-around for
/// the fact that the `align_of` intrinsic currently requires the
/// input type to be Sized (which I do not think is strictly
/// necessary).
#[unstable(feature = "placement_in", reason = "placement box design is still being worked out.")]
pub struct IntermediateBox<T: ?Sized>{
    ptr: *mut u8,
    size: uint,
    align: uint,
    marker: marker::PhantomData<*mut T>,
}

impl<T: ?Sized> Place<T> for IntermediateBox<T> {
    fn pointer(&mut self) -> *mut T { self.ptr as *mut T }
}

unsafe fn finalize<T>(b: IntermediateBox<T>) -> Box<T> {
    let p = b.ptr as *mut T;
    mem::forget(b);
    mem::transmute(p)
}

fn make_place<T>() -> IntermediateBox<T> {
    let size = mem::size_of::<T>();
    let align = mem::align_of::<T>();

    let p = if size == 0 {
        heap::EMPTY as *mut u8
    } else {
        let p = unsafe {
            heap::allocate(size, align)
        };
        if p.is_null() {
            panic!("Box make_place allocation failure.");
        }
        p
    };

    IntermediateBox { ptr: p, size: size, align: align, marker: marker::PhantomData }
}

impl<T> BoxPlace<T> for IntermediateBox<T> {
    fn make_place() -> IntermediateBox<T> { make_place() }
}

impl<T> InPlace<T> for IntermediateBox<T> {
    type Owner = Box<T>;
    unsafe fn finalize(self) -> Box<T> { finalize(self) }
}

impl<T> Boxed for Box<T> {
    type Data = T;
    type Place = IntermediateBox<T>;
    unsafe fn finalize(b: IntermediateBox<T>) -> Box<T> { finalize(b) }
}

impl<T> Placer<T> for ExchangeHeapSingleton {
    type Place = IntermediateBox<T>;

    fn make_place(self) -> IntermediateBox<T> {
        make_place()
    }
}

#[unsafe_destructor]
impl<T: ?Sized> Drop for IntermediateBox<T> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                heap::deallocate(self.ptr, self.size, self.align)
            }
        }
    }
}

impl<T> Box<T> {
    /// Allocates memory on the heap and then moves `x` into it.
    ///
    /// # Examples
    ///
    /// ```
    /// let x = Box::new(5);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    pub fn new(x: T) -> Box<T> {
        box x
    }
}

impl<T : ?Sized> Box<T> {
    /// Constructs a box from the raw pointer.
    ///
    /// After this function call, pointer is owned by resulting box.
    /// In particular, it means that `Box` destructor calls destructor
    /// of `T` and releases memory. Since the way `Box` allocates and
    /// releases memory is unspecified, the only valid pointer to pass
    /// to this function is the one taken from another `Box` with
    /// `boxed::into_raw` function.
    ///
    /// Function is unsafe, because improper use of this function may
    /// lead to memory problems like double-free, for example if the
    /// function is called twice on the same raw pointer.
    #[unstable(feature = "alloc",
               reason = "may be renamed or moved out of Box scope")]
    #[inline]
    pub unsafe fn from_raw(raw: *mut T) -> Self {
        mem::transmute(raw)
    }
}

/// Consumes the `Box`, returning the wrapped raw pointer.
///
/// After call to this function, caller is responsible for the memory
/// previously managed by `Box`, in particular caller should properly
/// destroy `T` and release memory. The proper way to do it is to
/// convert pointer back to `Box` with `Box::from_raw` function, because
/// `Box` does not specify, how memory is allocated.
///
/// Function is unsafe, because result of this function is no longer
/// automatically managed that may lead to memory or other resource
/// leak.
///
/// # Examples
/// ```
/// use std::boxed;
///
/// let seventeen = Box::new(17u32);
/// let raw = unsafe { boxed::into_raw(seventeen) };
/// let boxed_again = unsafe { Box::from_raw(raw) };
/// ```
#[unstable(feature = "alloc",
           reason = "may be renamed")]
#[inline]
pub unsafe fn into_raw<T : ?Sized>(b: Box<T>) -> *mut T {
    mem::transmute(b)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: Default> Default for Box<T> {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Box<T> { box Default::default() }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T> Default for Box<[T]> {
    #[stable(feature = "rust1", since = "1.0.0")]
    fn default() -> Box<[T]> { Box::<[T; 0]>::new([]) }
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
    #[inline]
    fn clone(&self) -> Box<T> { box (HEAP) {(**self).clone()} }
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

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { PartialEq::ne(&**self, &**other) }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized + PartialOrd> PartialOrd for Box<T> {
    #[inline]
    fn partial_cmp(&self, other: &Box<T>) -> Option<Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
    #[inline]
    fn lt(&self, other: &Box<T>) -> bool { PartialOrd::lt(&**self, &**other) }
    #[inline]
    fn le(&self, other: &Box<T>) -> bool { PartialOrd::le(&**self, &**other) }
    #[inline]
    fn ge(&self, other: &Box<T>) -> bool { PartialOrd::ge(&**self, &**other) }
    #[inline]
    fn gt(&self, other: &Box<T>) -> bool { PartialOrd::gt(&**self, &**other) }
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
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

/// Extension methods for an owning `Any` trait object.
#[unstable(feature = "alloc",
           reason = "this trait will likely disappear once compiler bugs blocking \
                     a direct impl on `Box<Any>` have been fixed ")]
// FIXME(#18737): this should be a direct impl on `Box<Any>`. If you're
//                removing this please make sure that you can downcase on
//                `Box<Any + Send>` as well as `Box<Any>`
pub trait BoxAny {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Self>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl BoxAny for Box<Any> {
    #[inline]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Box<Any>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let raw = into_raw(self);
                let to: TraitObject =
                    mem::transmute::<*mut Any, TraitObject>(raw);

                // Extract the data pointer
                Ok(Box::from_raw(to.data as *mut T))
            }
        } else {
            Err(self)
        }
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
impl fmt::Debug for Box<Any> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Box<Any>")
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T { &**self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T { &mut **self }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator + ?Sized> Iterator for Box<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<I::Item> { (**self).next() }
    fn size_hint(&self) -> (usize, Option<usize>) { (**self).size_hint() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: DoubleEndedIterator + ?Sized> DoubleEndedIterator for Box<I> {
    fn next_back(&mut self) -> Option<I::Item> { (**self).next_back() }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl<I: ExactSizeIterator + ?Sized> ExactSizeIterator for Box<I> {}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a, E: Error + 'a> FromError<E> for Box<Error + 'a> {
    fn from_error(err: E) -> Box<Error + 'a> {
        Box::new(err)
    }
}
