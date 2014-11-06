// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A unique pointer type.

use heap;

use core::any::{Any, AnyRefExt};
use core::clone::Clone;
use core::cmp::{PartialEq, PartialOrd, Eq, Ord, Ordering};
use core::default::Default;
use core::fmt;
use core::intrinsics;
use core::kinds::Sized;
use core::mem;
use core::ops::{Drop, Placer, PlacementAgent};
use core::option::Option;
use core::raw::TraitObject;
use core::result::{Ok, Err, Result};

/// A value that represents the global exchange heap. This is the default
/// place that the `box` keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
/// ```rust
/// use std::boxed::HEAP;
///
/// # struct Bar;
/// # impl Bar { fn new(_a: int) { } }
/// let foo = box(HEAP) Bar::new(2);
/// let foo = box Bar::new(2);
/// ```
#[lang = "exchange_heap"]
#[experimental = "may be renamed; uncertain about custom allocator design"]
pub const HEAP: ExchangeHeapSingleton =
    ExchangeHeapSingleton { _force_singleton: () };

/// This the singleton type used solely for `boxed::HEAP`.
pub struct ExchangeHeapSingleton { _force_singleton: () }

/// A type that represents a uniquely-owned value.
#[lang = "owned_box"]
#[unstable = "custom allocators will add an additional type parameter (with default)"]
pub struct Box<Sized? T>(*mut T);

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
/// parameter has the `Sized?` generalization marker, and is also why
/// this carries an explicit size. However, it probably does not need
/// to carry the explicit alignment; that is just a work-around for
/// the fact that the `align_of` intrinsic currently requires the
/// input type to be Sized (which I do not think is strictly
/// necessary).
pub struct IntermediateBox<Sized? T>{
    ptr: *mut u8,
    size: uint,
    align: uint,
}

impl<T> Placer<T, Box<T>, IntermediateBox<T>> for ExchangeHeapSingleton {
    fn make_place(&mut self) -> IntermediateBox<T> {
        let size = mem::size_of::<T>();
        let align = mem::align_of::<T>();

        let p = if size == 0 {
            heap::EMPTY as *mut u8
        } else {
            unsafe {
                heap::allocate(size, align)
            }
        };

        IntermediateBox { ptr: p, size: size, align: align }
    }
}

impl<Sized? T> PlacementAgent<T, Box<T>> for IntermediateBox<T> {
    unsafe fn pointer(&self) -> *mut T {
        self.ptr as *mut T
    }
    unsafe fn finalize(self) -> Box<T> {
        let p = self.ptr as *mut T;
        mem::forget(self);
        mem::transmute(p)
    }
}

#[unsafe_destructor]
impl<Sized? T> Drop for IntermediateBox<T> {
    fn drop(&mut self) {
        if self.size > 0 {
            unsafe {
                heap::deallocate(self.ptr as *mut u8, self.size, self.align)
            }
        }
    }
}

impl<T: Default> Default for Box<T> {
    fn default() -> Box<T> {
        box (HEAP) { let t : T = Default::default(); t }
    }
}

#[unstable]
impl<T: Clone> Clone for Box<T> {
    /// Returns a copy of the owned box.
    #[inline]
    fn clone(&self) -> Box<T> { box (HEAP) {(**self).clone()} }

    /// Performs copy-assignment from `source` by reusing the existing allocation.
    #[inline]
    fn clone_from(&mut self, source: &Box<T>) {
        (**self).clone_from(&(**source));
    }
}

impl<T:PartialEq> PartialEq for Box<T> {
    #[inline]
    fn eq(&self, other: &Box<T>) -> bool { *(*self) == *(*other) }
    #[inline]
    fn ne(&self, other: &Box<T>) -> bool { *(*self) != *(*other) }
}
impl<T:PartialOrd> PartialOrd for Box<T> {
    #[inline]
    fn partial_cmp(&self, other: &Box<T>) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
    #[inline]
    fn lt(&self, other: &Box<T>) -> bool { *(*self) < *(*other) }
    #[inline]
    fn le(&self, other: &Box<T>) -> bool { *(*self) <= *(*other) }
    #[inline]
    fn ge(&self, other: &Box<T>) -> bool { *(*self) >= *(*other) }
    #[inline]
    fn gt(&self, other: &Box<T>) -> bool { *(*self) > *(*other) }
}
impl<T: Ord> Ord for Box<T> {
    #[inline]
    fn cmp(&self, other: &Box<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}
impl<T: Eq> Eq for Box<T> {}

/// Extension methods for an owning `Any` trait object.
#[unstable = "post-DST and coherence changes, this will not be a trait but \
              rather a direct `impl` on `Box<Any>`"]
pub trait BoxAny {
    /// Returns the boxed value if it is of type `T`, or
    /// `Err(Self)` if it isn't.
    #[unstable = "naming conventions around accessing innards may change"]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Self>;
}

#[stable]
impl BoxAny for Box<Any+'static> {
    #[inline]
    fn downcast<T: 'static>(self) -> Result<Box<T>, Box<Any+'static>> {
        if self.is::<T>() {
            unsafe {
                // Get the raw representation of the trait object
                let to: TraitObject =
                    *mem::transmute::<&Box<Any>, &TraitObject>(&self);

                // Prevent destructor on self being run
                intrinsics::forget(self);

                // Extract the data pointer
                Ok(mem::transmute(to.data))
            }
        } else {
            Err(self)
        }
    }
}

impl<Sized? T: fmt::Show> fmt::Show for Box<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl fmt::Show for Box<Any+'static> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Box<Any>")
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_owned_clone() {
        let a = box 5i;
        let b: Box<int> = a.clone();
        assert!(a == b);
    }

    #[test]
    fn any_move() {
        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        match a.downcast::<uint>() {
            Ok(a) => { assert!(a == box 8u); }
            Err(..) => panic!()
        }
        match b.downcast::<Test>() {
            Ok(a) => { assert!(a == box Test); }
            Err(..) => panic!()
        }

        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;

        assert!(a.downcast::<Box<Test>>().is_err());
        assert!(b.downcast::<Box<uint>>().is_err());
    }

    #[test]
    fn test_show() {
        let a = box 8u as Box<Any>;
        let b = box Test as Box<Any>;
        let a_str = a.to_str();
        let b_str = b.to_str();
        assert_eq!(a_str.as_slice(), "Box<Any>");
        assert_eq!(b_str.as_slice(), "Box<Any>");

        let a = &8u as &Any;
        let b = &Test as &Any;
        let s = format!("{}", a);
        assert_eq!(s.as_slice(), "&Any");
        let s = format!("{}", b);
        assert_eq!(s.as_slice(), "&Any");
    }
}
