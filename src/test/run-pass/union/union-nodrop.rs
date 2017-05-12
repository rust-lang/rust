// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_intrinsics)]
#![feature(untagged_unions)]

#![allow(unions_with_drop_fields)]
#![allow(dead_code)]

use std::intrinsics::needs_drop;

// drop types in destructors should not require
// drop_types_in_const
static X: Option<NoDrop<Box<u8>>> = None;

// A union that scrubs the drop glue from its inner type
union NoDrop<T> {inner: T}

// Copy currently can't be implemented on drop-containing unions,
// this may change later
// https://github.com/rust-lang/rust/pull/38934#issuecomment-271219289

// // We should be able to implement Copy for NoDrop
// impl<T> Copy for NoDrop<T> {}
// impl<T> Clone for NoDrop<T> {fn clone(&self) -> Self { *self }}

// // We should be able to implement Copy for things using NoDrop
// #[derive(Copy, Clone)]
struct Foo {
    x: NoDrop<Box<u8>>
}

struct Baz {
    x: NoDrop<Box<u8>>,
    y: Box<u8>,
}

union ActuallyDrop<T> {inner: T}

impl<T> Drop for ActuallyDrop<T> {
    fn drop(&mut self) {}
}

fn main() {
    unsafe {
        // NoDrop should not make needs_drop true
        assert!(!needs_drop::<Foo>());
        assert!(!needs_drop::<NoDrop<u8>>());
        assert!(!needs_drop::<NoDrop<Box<u8>>>());
        // presence of other drop types should still work
        assert!(needs_drop::<Baz>());
        // drop impl on union itself should work
        assert!(needs_drop::<ActuallyDrop<u8>>());
        assert!(needs_drop::<ActuallyDrop<Box<u8>>>());
    }
}
