// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(missing_doc)]
#![experimental]

//! Contains struct definitions for the layout of compiler built-in types.
//!
//! They can be used as targets of transmutes in unsafe code for manipulating
//! the raw representations directly.
//!
//! Their definition should always match the ABI defined in `rustc::back::abi`.

use cast;

/// The representation of a Rust managed box
pub struct Box<T> {
    pub ref_count: uint,
    pub drop_glue: fn(ptr: *mut u8),
    pub prev: *mut Box<T>,
    pub next: *mut Box<T>,
    pub data: T,
}

/// The representation of a Rust vector
pub struct Vec<T> {
    pub fill: uint,
    pub alloc: uint,
    pub data: T,
}

/// The representation of a Rust string
pub type String = Vec<u8>;

/// The representation of a Rust slice
pub struct Slice<T> {
    pub data: *T,
    pub len: uint,
}

/// The representation of a Rust closure
pub struct Closure {
    pub code: *(),
    pub env: *(),
}

/// The representation of a Rust procedure (`proc()`)
pub struct Procedure {
    pub code: *(),
    pub env: *(),
}

/// The representation of a Rust trait object.
///
/// This struct does not have a `Repr` implementation
/// because there is no way to refer to all trait objects generically.
pub struct TraitObject {
    pub vtable: *(),
    pub data: *(),
}

/// This trait is meant to map equivalences between raw structs and their
/// corresponding rust values.
pub trait Repr<T> {
    /// This function "unwraps" a rust value (without consuming it) into its raw
    /// struct representation. This can be used to read/write different values
    /// for the struct. This is a safe method because by default it does not
    /// enable write-access to the fields of the return value in safe code.
    #[inline]
    fn repr(&self) -> T { unsafe { cast::transmute_copy(self) } }
}

impl<'a, T> Repr<Slice<T>> for &'a [T] {}
impl<'a> Repr<Slice<u8>> for &'a str {}
impl<T> Repr<*Box<T>> for @T {}
impl<T> Repr<*Vec<T>> for ~[T] {}
impl Repr<*String> for ~str {}

#[cfg(test)]
mod tests {
    use super::*;

    use cast;

    #[test]
    fn synthesize_closure() {
        unsafe {
            let x = 10;
            let f: |int| -> int = |y| x + y;

            assert_eq!(f(20), 30);

            let original_closure: Closure = cast::transmute(f);

            let actual_function_pointer = original_closure.code;
            let environment = original_closure.env;

            let new_closure = Closure {
                code: actual_function_pointer,
                env: environment
            };

            let new_f: |int| -> int = cast::transmute(new_closure);
            assert_eq!(new_f(20), 30);
        }
    }
}
