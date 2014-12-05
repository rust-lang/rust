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
#![experimental]

//! Contains struct definitions for the layout of compiler built-in types.
//!
//! They can be used as targets of transmutes in unsafe code for manipulating
//! the raw representations directly.
//!
//! Their definition should always match the ABI defined in `rustc::back::abi`.

use kinds::Copy;
use mem;
use kinds::Sized;

/// The representation of a Rust slice
#[repr(C)]
pub struct Slice<T> {
    pub data: *const T,
    pub len: uint,
}

impl<T> Copy for Slice<T> {}

/// The representation of a Rust closure
#[repr(C)]
pub struct Closure {
    pub code: *mut (),
    pub env: *mut (),
}

impl Copy for Closure {}

/// The representation of a Rust procedure (`proc()`)
#[repr(C)]
pub struct Procedure {
    pub code: *mut (),
    pub env: *mut (),
}

impl Copy for Procedure {}

/// The representation of a Rust trait object.
///
/// This struct does not have a `Repr` implementation
/// because there is no way to refer to all trait objects generically.
#[repr(C)]
pub struct TraitObject {
    pub data: *mut (),
    pub vtable: *mut (),
}

impl Copy for TraitObject {}

/// This trait is meant to map equivalences between raw structs and their
/// corresponding rust values.
pub trait Repr<T> for Sized? {
    /// This function "unwraps" a rust value (without consuming it) into its raw
    /// struct representation. This can be used to read/write different values
    /// for the struct. This is a safe method because by default it does not
    /// enable write-access to the fields of the return value in safe code.
    #[inline]
    fn repr(&self) -> T { unsafe { mem::transmute_copy(&self) } }
}

impl<T> Repr<Slice<T>> for [T] {}
impl Repr<Slice<u8>> for str {}
