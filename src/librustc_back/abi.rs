// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_uppercase_statics)]

pub const box_field_refcnt: uint = 0u;
pub const box_field_drop_glue: uint = 1u;
pub const box_field_body: uint = 4u;

pub const tydesc_field_visit_glue: uint = 3u;

// The two halves of a closure: code and environment.
pub const fn_field_code: uint = 0u;
pub const fn_field_box: uint = 1u;

// The two fields of a trait object/trait instance: vtable and box.
// The vtable contains the type descriptor as first element.
pub const trt_field_box: uint = 0u;
pub const trt_field_vtable: uint = 1u;

pub const slice_elt_base: uint = 0u;
pub const slice_elt_len: uint = 1u;
