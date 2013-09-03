// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub static rc_base_field_refcnt: uint = 0u;

pub static task_field_refcnt: uint = 0u;

pub static task_field_stk: uint = 2u;

pub static task_field_runtime_sp: uint = 3u;

pub static task_field_rust_sp: uint = 4u;

pub static task_field_gc_alloc_chain: uint = 5u;

pub static task_field_dom: uint = 6u;

pub static n_visible_task_fields: uint = 7u;

pub static dom_field_interrupt_flag: uint = 1u;

pub static frame_glue_fns_field_mark: uint = 0u;

pub static frame_glue_fns_field_drop: uint = 1u;

pub static frame_glue_fns_field_reloc: uint = 2u;

pub static box_field_refcnt: uint = 0u;
pub static box_field_tydesc: uint = 1u;
pub static box_field_prev: uint = 2u;
pub static box_field_next: uint = 3u;
pub static box_field_body: uint = 4u;

pub static general_code_alignment: uint = 16u;

pub static tydesc_field_size: uint = 0u;
pub static tydesc_field_align: uint = 1u;
pub static tydesc_field_take_glue: uint = 2u;
pub static tydesc_field_drop_glue: uint = 3u;
pub static tydesc_field_free_glue: uint = 4u;
pub static tydesc_field_visit_glue: uint = 5u;
pub static tydesc_field_borrow_offset: uint = 6u;
pub static tydesc_field_name_offset: uint = 7u;
pub static n_tydesc_fields: uint = 8u;

// The two halves of a closure: code and environment.
pub static fn_field_code: uint = 0u;
pub static fn_field_box: uint = 1u;

// The two fields of a trait object/trait instance: vtable and box.
// The vtable contains the type descriptor as first element.
pub static trt_field_vtable: uint = 0u;
pub static trt_field_box: uint = 1u;

pub static vec_elt_fill: uint = 0u;

pub static vec_elt_alloc: uint = 1u;

pub static vec_elt_elems: uint = 2u;

pub static slice_elt_base: uint = 0u;
pub static slice_elt_len: uint = 1u;

pub static abi_version: uint = 1u;
