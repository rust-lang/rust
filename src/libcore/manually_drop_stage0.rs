// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A wrapper to inhibit compiler from automatically calling `T`’s destructor.
///
/// This wrapper is 0-cost.
///
/// # Examples
///
/// This wrapper helps with explicitly documenting the drop order dependencies between fields of
/// the type:
///
/// ```rust
/// use std::mem::ManuallyDrop;
/// struct Peach;
/// struct Banana;
/// struct Melon;
/// struct FruitBox {
///     // Immediately clear there’s something non-trivial going on with these fields.
///     peach: ManuallyDrop<Peach>,
///     melon: Melon, // Field that’s independent of the other two.
///     banana: ManuallyDrop<Banana>,
/// }
///
/// impl Drop for FruitBox {
///     fn drop(&mut self) {
///         unsafe {
///             // Explicit ordering in which field destructors are run specified in the intuitive
///             // location – the destructor of the structure containing the fields.
///             // Moreover, one can now reorder fields within the struct however much they want.
///             ManuallyDrop::drop(&mut self.peach);
///             ManuallyDrop::drop(&mut self.banana);
///         }
///         // After destructor for `FruitBox` runs (this function), the destructor for Melon gets
///         // invoked in the usual manner, as it is not wrapped in `ManuallyDrop`.
///     }
/// }
/// ```
#[stable(feature = "manually_drop", since = "1.20.0")]
#[allow(unions_with_drop_fields)]
#[derive(Copy)]
pub union ManuallyDrop<T>{ value: T }

impl<T> ManuallyDrop<T> {
    /// Wrap a value to be manually dropped.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// ManuallyDrop::new(Box::new(()));
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[rustc_const_unstable(feature = "const_manually_drop_new")]
    #[inline]
    pub const fn new(value: T) -> ManuallyDrop<T> {
        ManuallyDrop { value: value }
    }

    /// Extract the value from the ManuallyDrop container.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use std::mem::ManuallyDrop;
    /// let x = ManuallyDrop::new(Box::new(()));
    /// let _: Box<()> = ManuallyDrop::into_inner(x);
    /// ```
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline]
    pub fn into_inner(slot: ManuallyDrop<T>) -> T {
        unsafe {
            slot.value
        }
    }

    /// Manually drops the contained value.
    ///
    /// # Safety
    ///
    /// This function runs the destructor of the contained value and thus the wrapped value
    /// now represents uninitialized data. It is up to the user of this method to ensure the
    /// uninitialized data is not actually used.
    #[stable(feature = "manually_drop", since = "1.20.0")]
    #[inline]
    pub unsafe fn drop(slot: &mut ManuallyDrop<T>) {
        ptr::drop_in_place(&mut slot.value)
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T> Deref for ManuallyDrop<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            &self.value
        }
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T> DerefMut for ManuallyDrop<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            &mut self.value
        }
    }
}

#[stable(feature = "manually_drop", since = "1.20.0")]
impl<T: ::fmt::Debug> ::fmt::Debug for ManuallyDrop<T> {
    fn fmt(&self, fmt: &mut ::fmt::Formatter) -> ::fmt::Result {
        unsafe {
            fmt.debug_tuple("ManuallyDrop").field(&self.value).finish()
        }
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: Clone> Clone for ManuallyDrop<T> {
    fn clone(&self) -> Self {
        ManuallyDrop::new(self.deref().clone())
    }

    fn clone_from(&mut self, source: &Self) {
        self.deref_mut().clone_from(source);
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: Default> Default for ManuallyDrop<T> {
    fn default() -> Self {
        ManuallyDrop::new(Default::default())
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: PartialEq> PartialEq for ManuallyDrop<T> {
    fn eq(&self, other: &Self) -> bool {
        self.deref().eq(other)
    }

    fn ne(&self, other: &Self) -> bool {
        self.deref().ne(other)
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: Eq> Eq for ManuallyDrop<T> {}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: PartialOrd> PartialOrd for ManuallyDrop<T> {
    fn partial_cmp(&self, other: &Self) -> Option<::cmp::Ordering> {
        self.deref().partial_cmp(other)
    }

    fn lt(&self, other: &Self) -> bool {
        self.deref().lt(other)
    }

    fn le(&self, other: &Self) -> bool {
        self.deref().le(other)
    }

    fn gt(&self, other: &Self) -> bool {
        self.deref().gt(other)
    }

    fn ge(&self, other: &Self) -> bool {
        self.deref().ge(other)
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: Ord> Ord for ManuallyDrop<T> {
    fn cmp(&self, other: &Self) -> ::cmp::Ordering {
        self.deref().cmp(other)
    }
}

#[stable(feature = "manually_drop_impls", since = "1.22.0")]
impl<T: ::hash::Hash> ::hash::Hash for ManuallyDrop<T> {
    fn hash<H: ::hash::Hasher>(&self, state: &mut H) {
        self.deref().hash(state);
    }
}
