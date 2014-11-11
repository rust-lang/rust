// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub const BOX_FIELD_DROP_GLUE: uint = 1u;
pub const BOX_FIELD_BODY: uint = 4u;

/// The first half of a fat pointer.
/// - For a closure, this is the code address.
/// - For an object or trait instance, this is the address of the box.
/// - For a slice, this is the base address.
pub const FAT_PTR_ADDR: uint = 0;

/// The second half of a fat pointer.
/// - For a closure, this is the address of the environment.
/// - For an object or trait instance, this is the address of the vtable.
/// - For a slice, this is the length.
pub const FAT_PTR_EXTRA: uint = 1u;
