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
//! `Box<T>`, casually referred to as a 'box', provides the simplest form of heap allocation in
//! Rust. Boxes provide ownership for this allocation, and drop their contents when they go out of
//! scope.
//!
//! Boxes are useful in two situations: recursive data structures, and occasionally when returning
//! data. [The Pointer chapter of the Book](../../../book/pointers.html#best-practices-1) explains
//! these cases in detail.
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
//! #[derive(Show)]
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
//! This will print `Cons(1i32, Box(Cons(2i32, Box(Nil))))`.

#![stable]

use heap;

use core::any::Any;
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::error::{Error, FromError};
use core::fmt;
use core::hash::{self, Hash};
use core::iter::Iterator;
use core::marker::{Copy, Sized};
use core::mem;
use core::ops::{Deref, DerefMut, Drop, Placer, PlacementAgent};
use core::option::Option;
use core::ptr::PtrExt;
use core::ptr::Unique;
use core::raw::TraitObject;
use core::result::Result::{Ok, Err};
use core::result::Result;

/// A value that represents the heap. This is the default place that the `box` keyword allocates
/// into when no place is supplied.
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
#[unstable = "may be renamed; uncertain about custom allocator design"]
pub const HEAP: ExchangeHeapSingleton =
    ExchangeHeapSingleton { _force_singleton: () };

/// This the singleton type used solely for `boxed::HEAP`.
pub struct ExchangeHeapSingleton { _force_singleton: () }
impl Copy for ExchangeHeapSingleton { }

/// A pointer type for heap allocation.
///
/// See the [module-level documentation](../../std/boxed/index.html) for more.
#[lang = "owned_box"]
#[stable]
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
#[experimental = "placement box design is still being worked out."]
pub struct IntermediateBox<T: ?Sized>{
    ptr: *mut u8,
    size: uint,
    align: uint,
}

impl<T: ?Sized> PlacementAgent<T> for IntermediateBox<T> {
    type Owner = Box<T>;

    fn pointer(&mut self) -> *mut T { self.ptr as *mut T }

    unsafe fn finalize(self) -> Box<T> {
        let p = self.ptr as *mut T;
        mem::forget(self);
        mem::transmute(p)
    }
}

impl<T> Placer<T> for ExchangeHeapSingleton {
    type Interim = IntermediateBox<T>;
    
    fn make_place(self) -> IntermediateBox<T> {
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

        IntermediateBox { ptr: p, size: size, align: align }
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
    #[stable]
    pub fn new(x: T) -> Box<T> {
        box x
    }
}

#[stable]
impl<T: Default> Default for Box<T> {
    #[stable]
    fn default() -> Box<T> { box (HEAP) Default::default() }
}

#[stable]
impl<T> Default for Box<[T]> {
    #[stable]
    fn default() -> Box<[T]> { box (HEAP) [] }
}

#[stable]
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

#[stable]
impl<T: ?Sized + PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { PartialEq::eq(&**self, &**other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { PartialEq::ne(&**self, &**other) }
}
#[stable]
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
#[stable]
impl<T: ?Sized + Ord> Ord for Box<T> {
    #[inline]
    fn cmp(&self, other: &Box<T>) -> Ordering {
        Ord::cmp(&**self, &**other)
    }
}
#[stable]
impl<T: ?Sized + Eq> Eq for Box<T> {}

impl<S: hash::Hasher, T: ?Sized + Hash<S>> Hash<S> for Box<T> {
    #[inline]
    fn hash(&self, state: &mut S) {
        (**self).hash(state);
    }
}

/// Extension methods for an owning `Any` trait object.
#[unstable = "this trait will likely disappear once compiler bugs blocking \
              a direct impl on `Box<Any>` have been fixed "]
// FIXME(#18737): this should be a direct impl on `Box<Any>`. If you're
//                removing this please make sure that you can downcase on
//                `Box<Any + Send>` as well as `Box<Any>`
pub trait BoxAny {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    #[stable]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Self>;
}

#[stable]
impl BoxAny for Box<Any> {
    #[inline]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Box<Any>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject =
                    mem::transmute::<Box<Any>, TraitObject>(self);

                // Extract the data pointer
                Ok(mem::transmute(to.data))
            }
        } else {
            Err(self)
        }
    }
}

#[stable]
impl<T: fmt::Display + ?Sized> fmt::Display for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

#[stable]
impl<T: fmt::Debug + ?Sized> fmt::Debug for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

#[stable]
impl fmt::Debug for Box<Any> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Box<Any>")
    }
}

#[stable]
impl<T: ?Sized> Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T { &**self }
}

#[stable]
impl<T: ?Sized> DerefMut for Box<T> {
    fn deref_mut(&mut self) -> &mut T { &mut **self }
}

// FIXME(#21363) remove `old_impl_check` when bug is fixed
#[old_impl_check]
impl<'a, T> Iterator for Box<Iterator<Item=T> + 'a> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        (**self).next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (**self).size_hint()
    }
}

impl<'a, E: Error + 'a> FromError<E> for Box<Error + 'a> {
    fn from_error(err: E) -> Box<Error + 'a> {
        Box::new(err)
    }
}
