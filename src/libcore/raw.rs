// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_docs)]
#![unstable(feature = "raw", issue = "27751")]

//! Contains struct definitions for the layout of compiler built-in types.
//!
//! They can be used as targets of transmutes in unsafe code for manipulating
//! the raw representations directly.
//!
//! Their definition should always match the ABI defined in `rustc::back::abi`.

use clone::Clone;
use marker::Copy;
use mem;

/// The representation of a slice like `&[T]`.
///
/// This struct is guaranteed to have the layout of types like `&[T]`,
/// `&str`, and `Box<[T]>`, but is not the type of such slices
/// (e.g. the fields are not directly accessible on a `&[T]`) nor does
/// it control that layout (changing the definition will not change
/// the layout of a `&[T]`). It is only designed to be used by unsafe
/// code that needs to manipulate the low-level details.
///
/// However, it is not recommended to use this type for such code,
/// since there are alternatives which may be safer:
///
/// - Creating a slice from a data pointer and length can be done with
///   `std::slice::from_raw_parts` or `std::slice::from_raw_parts_mut`
///   instead of `std::mem::transmute`ing a value of type `Slice`.
/// - Extracting the data pointer and length from a slice can be
///   performed with the `as_ptr` (or `as_mut_ptr`) and `len`
///   methods.
///
/// If one does decide to convert a slice value to a `Slice`, the
/// `Repr` trait in this module provides a method for a safe
/// conversion from `&[T]` (and `&str`) to a `Slice`, more type-safe
/// than a call to `transmute`.
///
/// # Examples
///
/// ```
/// #![feature(raw)]
///
/// use std::raw::{self, Repr};
///
/// let slice: &[u16] = &[1, 2, 3, 4];
///
/// let repr: raw::Slice<u16> = slice.repr();
/// println!("data pointer = {:?}, length = {}", repr.data, repr.len);
/// ```
#[repr(C)]
pub struct Slice<T> {
    pub data: *const T,
    pub len: usize,
}

impl<T> Copy for Slice<T> {}
impl<T> Clone for Slice<T> {
    fn clone(&self) -> Slice<T> { *self }
}

/// The representation of a trait object like `&SomeTrait`.
///
/// This struct has the same layout as types like `&SomeTrait` and
/// `Box<AnotherTrait>`. The [Trait Objects chapter of the
/// Book][moreinfo] contains more details about the precise nature of
/// these internals.
///
/// [moreinfo]: ../../book/trait-objects.html#representation
///
/// `TraitObject` is guaranteed to match layouts, but it is not the
/// type of trait objects (e.g. the fields are not directly accessible
/// on a `&SomeTrait`) nor does it control that layout (changing the
/// definition will not change the layout of a `&SomeTrait`). It is
/// only designed to be used by unsafe code that needs to manipulate
/// the low-level details.
///
/// There is no `Repr` implementation for `TraitObject` because there
/// is no way to refer to all trait objects generically, so the only
/// way to create values of this type is with functions like
/// `std::mem::transmute`. Similarly, the only way to create a true
/// trait object from a `TraitObject` value is with `transmute`.
///
/// Synthesizing a trait object with mismatched types—one where the
/// vtable does not correspond to the type of the value to which the
/// data pointer points—is highly likely to lead to undefined
/// behavior.
///
/// # Examples
///
/// ```
/// #![feature(raw)]
///
/// use std::mem;
/// use std::raw;
///
/// // an example trait
/// trait Foo {
///     fn bar(&self) -> i32;
/// }
/// impl Foo for i32 {
///     fn bar(&self) -> i32 {
///          *self + 1
///     }
/// }
///
/// let value: i32 = 123;
///
/// // let the compiler make a trait object
/// let object: &Foo = &value;
///
/// // look at the raw representation
/// let raw_object: raw::TraitObject = unsafe { mem::transmute(object) };
///
/// // the data pointer is the address of `value`
/// assert_eq!(raw_object.data as *const i32, &value as *const _);
///
///
/// let other_value: i32 = 456;
///
/// // construct a new object, pointing to a different `i32`, being
/// // careful to use the `i32` vtable from `object`
/// let synthesized: &Foo = unsafe {
///      mem::transmute(raw::TraitObject {
///          data: &other_value as *const _ as *mut (),
///          vtable: raw_object.vtable
///      })
/// };
///
/// // it should work just like we constructed a trait object out of
/// // `other_value` directly
/// assert_eq!(synthesized.bar(), 457);
/// ```
#[repr(C)]
#[derive(Copy, Clone)]
pub struct TraitObject {
    pub data: *mut (),
    pub vtable: *mut (),
}

/// This trait is meant to map equivalences between raw structs and their
/// corresponding rust values.
pub unsafe trait Repr<T> {
    /// This function "unwraps" a rust value (without consuming it) into its raw
    /// struct representation. This can be used to read/write different values
    /// for the struct. This is a safe method because by default it does not
    /// enable write-access to the fields of the return value in safe code.
    #[inline]
    fn repr(&self) -> T { unsafe { mem::transmute_copy(&self) } }
}

unsafe impl<T> Repr<Slice<T>> for [T] {}
unsafe impl Repr<Slice<u8>> for str {}
