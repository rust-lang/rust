// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static box_field_refcnt: uint = 0u;
pub static box_field_tydesc: uint = 1u;
pub static box_field_body: uint = 4u;

pub static tydesc_field_visit_glue: uint = 3u;

// The two halves of a closure: code and environment.
pub static fn_field_code: uint = 0u;
pub static fn_field_box: uint = 1u;

// The two fields of a trait object/trait instance: vtable and box.
// The vtable contains the type descriptor as first element.
pub static trt_field_box: uint = 0u;
pub static trt_field_vtable: uint = 1u;

pub static slice_elt_base: uint = 0u;
pub static slice_elt_len: uint = 1u;
