// Copyright 2013-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Private macro to get the offset of a struct field in bytes from the address of the struct.
macro_rules! offset_of {
    ($container:path, $field:ident) => {{
        // Make sure the field actually exists. This line ensures that a compile-time error is
        // generated if $field is accessed through a Deref impl.
        let $container { $field : _, .. };

        // Create an (invalid) instance of the container and calculate the offset to its
        // field. Using a null pointer might be UB if `&(*(0 as *const T)).field` is interpreted to
        // be nullptr deref.
        let invalid: $container = ::core::mem::uninitialized();
        let offset = &invalid.$field as *const _ as usize - &invalid as *const _ as usize;

        // Do not run destructors on the made up invalid instance.
        ::core::mem::forget(invalid);
        offset as isize
    }};
}
