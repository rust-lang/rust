// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
#[legacy_modes];

use core::bool;
use intrinsic::{TyDesc, get_tydesc, visit_tydesc, TyVisitor};
use libc::c_void;
use vec::UnboxedVecRepr;

#[doc = "High-level interfaces to `intrinsic::visit_ty` reflection system."]

/// Trait for visitor that wishes to reflect on data.
trait movable_ptr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void);
}

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

enum ptr_visit_adaptor<V: TyVisitor movable_ptr> = {
    inner: V
};

impl<V: TyVisitor movable_ptr> ptr_visit_adaptor<V> {

    #[inline(always)]
    fn bump(sz: uint) {
      do self.inner.move_ptr() |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    fn align(a: uint) {
      do self.inner.move_ptr() |p| {
            align(p as uint, a) as *c_void
      };
    }

    #[inline(always)]
    fn align_to<T>() {
        self.align(sys::min_align_of::<T>());
    }

    #[inline(always)]
    fn bump_past<T>() {
        self.bump(sys::size_of::<T>());
    }

}

impl<V: TyVisitor movable_ptr> ptr_visit_adaptor<V>: TyVisitor {

    fn visit_bot() -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_bot() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_nil() -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_nil() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_bool() -> bool {
        self.align_to::<bool>();
        if ! self.inner.visit_bool() { return false; }
        self.bump_past::<bool>();
        true
    }

    fn visit_int() -> bool {
        self.align_to::<int>();
        if ! self.inner.visit_int() { return false; }
        self.bump_past::<int>();
        true
    }

    fn visit_i8() -> bool {
        self.align_to::<i8>();
        if ! self.inner.visit_i8() { return false; }
        self.bump_past::<i8>();
        true
    }

    fn visit_i16() -> bool {
        self.align_to::<i16>();
        if ! self.inner.visit_i16() { return false; }
        self.bump_past::<i16>();
        true
    }

    fn visit_i32() -> bool {
        self.align_to::<i32>();
        if ! self.inner.visit_i32() { return false; }
        self.bump_past::<i32>();
        true
    }

    fn visit_i64() -> bool {
        self.align_to::<i64>();
        if ! self.inner.visit_i64() { return false; }
        self.bump_past::<i64>();
        true
    }

    fn visit_uint() -> bool {
        self.align_to::<uint>();
        if ! self.inner.visit_uint() { return false; }
        self.bump_past::<uint>();
        true
    }

    fn visit_u8() -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_u8() { return false; }
        self.bump_past::<u8>();
        true
    }

    fn visit_u16() -> bool {
        self.align_to::<u16>();
        if ! self.inner.visit_u16() { return false; }
        self.bump_past::<u16>();
        true
    }

    fn visit_u32() -> bool {
        self.align_to::<u32>();
        if ! self.inner.visit_u32() { return false; }
        self.bump_past::<u32>();
        true
    }

    fn visit_u64() -> bool {
        self.align_to::<u64>();
        if ! self.inner.visit_u64() { return false; }
        self.bump_past::<u64>();
        true
    }

    fn visit_float() -> bool {
        self.align_to::<float>();
        if ! self.inner.visit_float() { return false; }
        self.bump_past::<float>();
        true
    }

    fn visit_f32() -> bool {
        self.align_to::<f32>();
        if ! self.inner.visit_f32() { return false; }
        self.bump_past::<f32>();
        true
    }

    fn visit_f64() -> bool {
        self.align_to::<f64>();
        if ! self.inner.visit_f64() { return false; }
        self.bump_past::<f64>();
        true
    }

    fn visit_char() -> bool {
        self.align_to::<char>();
        if ! self.inner.visit_char() { return false; }
        self.bump_past::<char>();
        true
    }

    fn visit_str() -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_str() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_box() -> bool {
        self.align_to::<@str>();
        if ! self.inner.visit_estr_box() { return false; }
        self.bump_past::<@str>();
        true
    }

    fn visit_estr_uniq() -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_estr_uniq() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_slice() -> bool {
        self.align_to::<&static/str>();
        if ! self.inner.visit_estr_slice() { return false; }
        self.bump_past::<&static/str>();
        true
    }

    fn visit_estr_fixed(n: uint,
                        sz: uint,
                        align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_estr_fixed(n, sz, align) { return false; }
        self.bump(sz);
        true
    }

    fn visit_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_box(mtbl, inner) { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_ptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_ptr(mtbl, inner) { return false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_rptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&static/u8>();
        if ! self.inner.visit_rptr(mtbl, inner) { return false; }
        self.bump_past::<&static/u8>();
        true
    }

    fn visit_unboxed_vec(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<UnboxedVecRepr>();
        // FIXME (#3732): Inner really has to move its own pointers on this one.
        // or else possibly we could have some weird interface wherein we
        // read-off a word from inner's pointers, but the read-word has to
        // always be the same in all sub-pointers? Dubious.
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        true
    }

    fn visit_vec(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_evec_box(mtbl, inner) { return false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_evec_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_evec_uniq(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_slice(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&static/[u8]>();
        if ! self.inner.visit_evec_slice(mtbl, inner) { return false; }
        self.bump_past::<&static/[u8]>();
        true
    }

    fn visit_evec_fixed(n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        self.align(align);
        if ! self.inner.visit_evec_fixed(n, sz, align, mtbl, inner) {
            return false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_rec_field(i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_rec_field(i, name, mtbl, inner) { return false; }
        true
    }

    fn visit_leave_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_class(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_class_field(i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_class_field(i, name, mtbl, inner) {
            return false;
        }
        true
    }

    fn visit_leave_class(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_enter_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_tup_field(i: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_tup_field(i, inner) { return false; }
        true
    }

    fn visit_leave_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            return false
        }
        true
    }

    fn visit_fn_input(i: uint, mode: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_input(i, mode, inner) { return false; }
        true
    }

    fn visit_fn_output(retstyle: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_output(retstyle, inner) { return false; }
        true
    }

    fn visit_leave_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn(purity, proto, n_inputs, retstyle) {
            return false;
        }
        true
    }

    fn visit_enter_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_enum(n_variants, sz, align) { return false; }
        true
    }

    fn visit_enter_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_enum_variant_field(i: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_enum_variant_field(i, inner) { return false; }
        true
    }

    fn visit_leave_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_leave_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_enum(n_variants, sz, align) { return false; }
        true
    }

    fn visit_trait() -> bool {
        self.align_to::<TyVisitor>();
        if ! self.inner.visit_trait() { return false; }
        self.bump_past::<TyVisitor>();
        true
    }

    fn visit_var() -> bool {
        if ! self.inner.visit_var() { return false; }
        true
    }

    fn visit_var_integral() -> bool {
        if ! self.inner.visit_var_integral() { return false; }
        true
    }

    fn visit_param(i: uint) -> bool {
        if ! self.inner.visit_param(i) { return false; }
        true
    }

    fn visit_self() -> bool {
        self.align_to::<&static/u8>();
        if ! self.inner.visit_self() { return false; }
        self.align_to::<&static/u8>();
        true
    }

    fn visit_type() -> bool {
        if ! self.inner.visit_type() { return false; }
        true
    }

    fn visit_opaque_box() -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_opaque_box() { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_constr(inner: *TyDesc) -> bool {
        if ! self.inner.visit_constr(inner) { return false; }
        true
    }

    fn visit_closure_ptr(ck: uint) -> bool {
        self.align_to::<fn@()>();
        if ! self.inner.visit_closure_ptr(ck) { return false; }
        self.bump_past::<fn@()>();
        true
    }
}

enum my_visitor = @{
    mut ptr1: *c_void,
    mut ptr2: *c_void,
    mut vals: ~[~str]
};

impl my_visitor {
    fn get<T>(f: fn(T)) {
        unsafe {
            f(*(self.ptr1 as *T));
        }
    }

    fn visit_inner(inner: *TyDesc) -> bool {
        let u = my_visitor(*self);
        let v = ptr_visit_adaptor({inner: u});
        visit_tydesc(inner, v as TyVisitor);
        true
    }
}

impl my_visitor: movable_ptr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void) {
        self.ptr1 = adjustment(self.ptr1);
        self.ptr2 = adjustment(self.ptr2);
    }
}

impl my_visitor: TyVisitor {

    fn visit_bot() -> bool { true }
    fn visit_nil() -> bool { true }
    fn visit_bool() -> bool {
      do self.get::<bool>() |b| {
            self.vals += ~[bool::to_str(b)];
      };
      true
    }
    fn visit_int() -> bool {
      do self.get::<int>() |i| {
            self.vals += ~[int::to_str(i, 10u)];
      };
      true
    }
    fn visit_i8() -> bool { true }
    fn visit_i16() -> bool { true }
    fn visit_i32() -> bool { true }
    fn visit_i64() -> bool { true }

    fn visit_uint() -> bool { true }
    fn visit_u8() -> bool { true }
    fn visit_u16() -> bool { true }
    fn visit_u32() -> bool { true }
    fn visit_u64() -> bool { true }

    fn visit_float() -> bool { true }
    fn visit_f32() -> bool { true }
    fn visit_f64() -> bool { true }

    fn visit_char() -> bool { true }
    fn visit_str() -> bool { true }

    fn visit_estr_box() -> bool { true }
    fn visit_estr_uniq() -> bool { true }
    fn visit_estr_slice() -> bool { true }
    fn visit_estr_fixed(_n: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_vec(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_slice(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(_n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(_i: uint, _name: &str,
                       _mtbl: uint, inner: *TyDesc) -> bool {
        error!("rec field!");
        self.visit_inner(inner)
    }
    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(_i: uint, _name: &str,
                         _mtbl: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(_i: uint, inner: *TyDesc) -> bool {
        error!("tup field!");
        self.visit_inner(inner)
    }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        // FIXME (#3732): this needs to rewind between enum variants, or something.
        true
    }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(_i: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(_i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(_retstyle: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait() -> bool { true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }
    fn visit_opaque_box() -> bool { true }
    fn visit_constr(_inner: *TyDesc) -> bool { true }
    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

fn get_tydesc_for<T>(&&_t: T) -> *TyDesc {
    get_tydesc::<T>()
}

fn main() {
    let r = (1,2,3,true,false,{x:5,y:4,z:3});
    let p = ptr::addr_of(&r) as *c_void;
    let u = my_visitor(@{mut ptr1: p,
                         mut ptr2: p,
                         mut vals: ~[]});
    let v = ptr_visit_adaptor({inner: u});
    let td = get_tydesc_for(r);
    unsafe { error!("tydesc sz: %u, align: %u",
                    (*td).size, (*td).align); }
    let v = v as TyVisitor;
    visit_tydesc(td, v);

    for (copy u.vals).each |s| {
        io::println(fmt!("val: %s", *s));
    }
    error!("%?", copy u.vals);
    assert u.vals == ~[~"1", ~"2", ~"3", ~"true", ~"false", ~"5", ~"4", ~"3"];
 }
