// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use cast;

/// The representation of a Rust managed box
pub struct Box<T> {
    ref_count: uint,
    drop_glue: fn(ptr: *mut u8),
    prev: *mut Box<T>,
    next: *mut Box<T>,
    data: T
}

/// The representation of a Rust vector
pub struct Vec<T> {
    fill: uint,
    alloc: uint,
    data: T
}

/// The representation of a Rust string
pub type String = Vec<u8>;

/// The representation of a Rust slice
pub struct Slice<T> {
    data: *T,
    len: uint
}

/// The representation of a Rust closure
pub struct Closure {
    code: *(),
    env: *(),
}

/// The representation of a Rust procedure (`proc()`)
pub struct Procedure {
    code: *(),
    env: *(),
}

/// This trait is meant to map equivalences between raw structs and their
/// corresponding rust values.
pub trait Repr<T> {
    /// This function "unwraps" a rust value (without consuming it) into its raw
    /// struct representation. This can be used to read/write different values
    /// for the struct. This is a safe method because by default it does not
    /// give write-access to the struct returned.
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
