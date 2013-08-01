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

use std::int;
use std::libc::c_void;
use std::ptr;
use std::sys;
use std::unstable::intrinsics::{TyDesc, get_tydesc, visit_tydesc, TyVisitor, Opaque};
use std::unstable::raw::Vec;

#[doc = "High-level interfaces to `std::unstable::intrinsics::visit_ty` reflection system."]

/// Trait for visitor that wishes to reflect on data.
trait movable_ptr {
    fn move_ptr(&self, adjustment: &fn(*c_void) -> *c_void);
}

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

struct ptr_visit_adaptor<V>(Inner<V>);

impl<V:TyVisitor + movable_ptr> ptr_visit_adaptor<V> {

    #[inline(always)]
    pub fn bump(&self, sz: uint) {
      do self.inner.move_ptr() |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    pub fn align(&self, a: uint) {
      do self.inner.move_ptr() |p| {
            align(p as uint, a) as *c_void
      };
    }

    #[inline(always)]
    pub fn align_to<T>(&self) {
        self.align(sys::min_align_of::<T>());
    }

    #[inline(always)]
    pub fn bump_past<T>(&self) {
        self.bump(sys::size_of::<T>());
    }

}

impl<V:TyVisitor + movable_ptr> TyVisitor for ptr_visit_adaptor<V> {

    fn visit_bot(&self) -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_bot() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_nil(&self) -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_nil() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_bool(&self) -> bool {
        self.align_to::<bool>();
        if ! self.inner.visit_bool() { return false; }
        self.bump_past::<bool>();
        true
    }

    fn visit_int(&self) -> bool {
        self.align_to::<int>();
        if ! self.inner.visit_int() { return false; }
        self.bump_past::<int>();
        true
    }

    fn visit_i8(&self) -> bool {
        self.align_to::<i8>();
        if ! self.inner.visit_i8() { return false; }
        self.bump_past::<i8>();
        true
    }

    fn visit_i16(&self) -> bool {
        self.align_to::<i16>();
        if ! self.inner.visit_i16() { return false; }
        self.bump_past::<i16>();
        true
    }

    fn visit_i32(&self) -> bool {
        self.align_to::<i32>();
        if ! self.inner.visit_i32() { return false; }
        self.bump_past::<i32>();
        true
    }

    fn visit_i64(&self) -> bool {
        self.align_to::<i64>();
        if ! self.inner.visit_i64() { return false; }
        self.bump_past::<i64>();
        true
    }

    fn visit_uint(&self) -> bool {
        self.align_to::<uint>();
        if ! self.inner.visit_uint() { return false; }
        self.bump_past::<uint>();
        true
    }

    fn visit_u8(&self) -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_u8() { return false; }
        self.bump_past::<u8>();
        true
    }

    fn visit_u16(&self) -> bool {
        self.align_to::<u16>();
        if ! self.inner.visit_u16() { return false; }
        self.bump_past::<u16>();
        true
    }

    fn visit_u32(&self) -> bool {
        self.align_to::<u32>();
        if ! self.inner.visit_u32() { return false; }
        self.bump_past::<u32>();
        true
    }

    fn visit_u64(&self) -> bool {
        self.align_to::<u64>();
        if ! self.inner.visit_u64() { return false; }
        self.bump_past::<u64>();
        true
    }

    fn visit_float(&self) -> bool {
        self.align_to::<float>();
        if ! self.inner.visit_float() { return false; }
        self.bump_past::<float>();
        true
    }

    fn visit_f32(&self) -> bool {
        self.align_to::<f32>();
        if ! self.inner.visit_f32() { return false; }
        self.bump_past::<f32>();
        true
    }

    fn visit_f64(&self) -> bool {
        self.align_to::<f64>();
        if ! self.inner.visit_f64() { return false; }
        self.bump_past::<f64>();
        true
    }

    fn visit_char(&self) -> bool {
        self.align_to::<char>();
        if ! self.inner.visit_char() { return false; }
        self.bump_past::<char>();
        true
    }

    fn visit_estr_box(&self) -> bool {
        self.align_to::<@str>();
        if ! self.inner.visit_estr_box() { return false; }
        self.bump_past::<@str>();
        true
    }

    fn visit_estr_uniq(&self) -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_estr_uniq() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_slice(&self) -> bool {
        self.align_to::<&'static str>();
        if ! self.inner.visit_estr_slice() { return false; }
        self.bump_past::<&'static str>();
        true
    }

    fn visit_estr_fixed(&self, n: uint,
                        sz: uint,
                        align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_estr_fixed(n, sz, align) { return false; }
        self.bump(sz);
        true
    }

    fn visit_box(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_box(mtbl, inner) { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_uniq_managed(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq_managed(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_ptr(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_ptr(mtbl, inner) { return false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_rptr(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&'static u8>();
        if ! self.inner.visit_rptr(mtbl, inner) { return false; }
        self.bump_past::<&'static u8>();
        true
    }

    fn visit_unboxed_vec(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<Vec<()>>();
        // FIXME (#3732): Inner really has to move its own pointers on this one.
        // or else possibly we could have some weird interface wherein we
        // read-off a word from inner's pointers, but the read-word has to
        // always be the same in all sub-pointers? Dubious.
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        true
    }

    fn visit_vec(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_box(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_evec_box(mtbl, inner) { return false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_evec_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_evec_uniq(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_uniq_managed(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[@u8]>();
        if ! self.inner.visit_evec_uniq_managed(mtbl, inner) { return false; }
        self.bump_past::<~[@u8]>();
        true
    }

    fn visit_evec_slice(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&'static [u8]>();
        if ! self.inner.visit_evec_slice(mtbl, inner) { return false; }
        self.bump_past::<&'static [u8]>();
        true
    }

    fn visit_evec_fixed(&self, n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        self.align(align);
        if ! self.inner.visit_evec_fixed(n, sz, align, mtbl, inner) {
            return false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_rec(&self, n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_rec_field(&self, i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_rec_field(i, name, mtbl, inner) { return false; }
        true
    }

    fn visit_leave_rec(&self, n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_class(&self, n_fields: uint, sz: uint, align: uint)
                      -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_class_field(&self, i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_class_field(i, name, mtbl, inner) {
            return false;
        }
        true
    }

    fn visit_leave_class(&self, n_fields: uint, sz: uint, align: uint)
                      -> bool {
        if ! self.inner.visit_leave_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_enter_tup(&self, n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_tup_field(&self, i: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_tup_field(i, inner) { return false; }
        true
    }

    fn visit_leave_tup(&self, n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_fn(&self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            return false
        }
        true
    }

    fn visit_fn_input(&self, i: uint, mode: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_input(i, mode, inner) { return false; }
        true
    }

    fn visit_fn_output(&self, retstyle: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_output(retstyle, inner) { return false; }
        true
    }

    fn visit_leave_fn(&self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn(purity, proto, n_inputs, retstyle) {
            return false;
        }
        true
    }

    fn visit_enter_enum(&self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint)
                     -> bool {
        self.align(align);
        if ! self.inner.visit_enter_enum(n_variants, get_disr, sz, align) { return false; }
        true
    }

    fn visit_enter_enum_variant(&self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_enum_variant_field(&self, i: uint, offset: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_enum_variant_field(i, offset, inner) { return false; }
        true
    }

    fn visit_leave_enum_variant(&self, variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_leave_enum(&self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        sz: uint, align: uint)
                     -> bool {
        if ! self.inner.visit_leave_enum(n_variants, get_disr, sz, align) { return false; }
        true
    }

    fn visit_trait(&self) -> bool {
        self.align_to::<@TyVisitor>();
        if ! self.inner.visit_trait() { return false; }
        self.bump_past::<@TyVisitor>();
        true
    }

    fn visit_var(&self) -> bool {
        if ! self.inner.visit_var() { return false; }
        true
    }

    fn visit_var_integral(&self) -> bool {
        if ! self.inner.visit_var_integral() { return false; }
        true
    }

    fn visit_param(&self, i: uint) -> bool {
        if ! self.inner.visit_param(i) { return false; }
        true
    }

    fn visit_self(&self) -> bool {
        self.align_to::<&'static u8>();
        if ! self.inner.visit_self() { return false; }
        self.align_to::<&'static u8>();
        true
    }

    fn visit_type(&self) -> bool {
        if ! self.inner.visit_type() { return false; }
        true
    }

    fn visit_opaque_box(&self) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_opaque_box() { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_constr(&self, inner: *TyDesc) -> bool {
        if ! self.inner.visit_constr(inner) { return false; }
        true
    }

    fn visit_closure_ptr(&self, ck: uint) -> bool {
        self.align_to::<@fn()>();
        if ! self.inner.visit_closure_ptr(ck) { return false; }
        self.bump_past::<@fn()>();
        true
    }
}

struct my_visitor(@mut Stuff);

struct Stuff {
    ptr1: *c_void,
    ptr2: *c_void,
    vals: ~[~str]
}

impl my_visitor {
    pub fn get<T:Clone>(&self, f: &fn(T)) {
        unsafe {
            f((*(self.ptr1 as *T)).clone());
        }
    }

    pub fn visit_inner(&self, inner: *TyDesc) -> bool {
        unsafe {
            let u = my_visitor(**self);
            let v = ptr_visit_adaptor::<my_visitor>(Inner {inner: u});
            visit_tydesc(inner, @v as @TyVisitor);
            true
        }
    }
}

struct Inner<V> { inner: V }

impl movable_ptr for my_visitor {
    fn move_ptr(&self, adjustment: &fn(*c_void) -> *c_void) {
        self.ptr1 = adjustment(self.ptr1);
        self.ptr2 = adjustment(self.ptr2);
    }
}

impl TyVisitor for my_visitor {

    fn visit_bot(&self) -> bool { true }
    fn visit_nil(&self) -> bool { true }
    fn visit_bool(&self) -> bool {
        do self.get::<bool>() |b| {
            self.vals.push(b.to_str());
        };
        true
    }
    fn visit_int(&self) -> bool {
        do self.get::<int>() |i| {
            self.vals.push(int::to_str(i));
        };
        true
    }
    fn visit_i8(&self) -> bool { true }
    fn visit_i16(&self) -> bool { true }
    fn visit_i32(&self) -> bool { true }
    fn visit_i64(&self) -> bool { true }

    fn visit_uint(&self) -> bool { true }
    fn visit_u8(&self) -> bool { true }
    fn visit_u16(&self) -> bool { true }
    fn visit_u32(&self) -> bool { true }
    fn visit_u64(&self) -> bool { true }

    fn visit_float(&self) -> bool { true }
    fn visit_f32(&self) -> bool { true }
    fn visit_f64(&self) -> bool { true }

    fn visit_char(&self) -> bool { true }

    fn visit_estr_box(&self) -> bool { true }
    fn visit_estr_uniq(&self) -> bool { true }
    fn visit_estr_slice(&self) -> bool { true }
    fn visit_estr_fixed(&self, _n: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq_managed(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_vec(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq_managed(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_slice(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(&self, _n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(&self, _i: uint, _name: &str,
                       _mtbl: uint, inner: *TyDesc) -> bool {
        error!("rec field!");
        self.visit_inner(inner)
    }
    fn visit_leave_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(&self, _i: uint, _name: &str,
                         _mtbl: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(&self, _i: uint, inner: *TyDesc) -> bool {
        error!("tup field!");
        self.visit_inner(inner)
    }
    fn visit_leave_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(&self, _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint, _align: uint) -> bool {
        // FIXME (#3732): this needs to rewind between enum variants, or something.
        true
    }
    fn visit_enter_enum_variant(&self, _variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(&self, _i: uint, _offset: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_enum_variant(&self, _variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(&self, _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(&self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(&self, _i: uint, _mode: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_fn_output(&self, _retstyle: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_leave_fn(&self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait(&self) -> bool { true }
    fn visit_var(&self) -> bool { true }
    fn visit_var_integral(&self) -> bool { true }
    fn visit_param(&self, _i: uint) -> bool { true }
    fn visit_self(&self) -> bool { true }
    fn visit_type(&self) -> bool { true }
    fn visit_opaque_box(&self) -> bool { true }
    fn visit_constr(&self, _inner: *TyDesc) -> bool { true }
    fn visit_closure_ptr(&self, _ck: uint) -> bool { true }
}

fn get_tydesc_for<T>(_t: T) -> *TyDesc {
    unsafe {
        get_tydesc::<T>()
    }
}

struct Triple { x: int, y: int, z: int }

pub fn main() {
    unsafe {
        let r = (1,2,3,true,false, Triple {x:5,y:4,z:3}, (12,));
        let p = ptr::to_unsafe_ptr(&r) as *c_void;
        let u = my_visitor(@mut Stuff {ptr1: p,
                                       ptr2: p,
                                       vals: ~[]});
        let v = ptr_visit_adaptor(Inner {inner: u});
        let td = get_tydesc_for(r);
        error!("tydesc sz: %u, align: %u",
               (*td).size, (*td).align);
        let v = @v as @TyVisitor;
        visit_tydesc(td, v);

        let r = u.vals.clone();
        foreach s in r.iter() {
            printfln!("val: %s", *s);
        }
        error!("%?", u.vals.clone());
        assert_eq!(u.vals.clone(),
                   ~[ ~"1", ~"2", ~"3", ~"true", ~"false", ~"5", ~"4", ~"3", ~"12"]);
    }
}
