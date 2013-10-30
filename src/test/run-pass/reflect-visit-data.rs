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

#[feature(managed_boxes)];

use std::libc::c_void;
use std::ptr;
use std::mem;
use std::unstable::intrinsics::{TyDesc, get_tydesc, visit_tydesc, TyVisitor, Disr, Opaque};
use std::unstable::raw::Vec;

#[doc = "High-level interfaces to `std::unstable::intrinsics::visit_ty` reflection system."]

/// Trait for visitor that wishes to reflect on data.
trait movable_ptr {
    fn move_ptr(&mut self, adjustment: &fn(*c_void) -> *c_void);
}

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

struct ptr_visit_adaptor<V>(Inner<V>);

impl<V:TyVisitor + movable_ptr> ptr_visit_adaptor<V> {

    #[inline(always)]
    pub fn bump(&mut self, sz: uint) {
      do self.inner.move_ptr() |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    pub fn align(&mut self, a: uint) {
      do self.inner.move_ptr() |p| {
            align(p as uint, a) as *c_void
      };
    }

    #[inline(always)]
    pub fn align_to<T>(&mut self) {
        self.align(mem::min_align_of::<T>());
    }

    #[inline(always)]
    pub fn bump_past<T>(&mut self) {
        self.bump(mem::size_of::<T>());
    }

}

impl<V:TyVisitor + movable_ptr> TyVisitor for ptr_visit_adaptor<V> {

    fn visit_bot(&mut self) -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_bot() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_nil(&mut self) -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_nil() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_bool(&mut self) -> bool {
        self.align_to::<bool>();
        if ! self.inner.visit_bool() { return false; }
        self.bump_past::<bool>();
        true
    }

    fn visit_int(&mut self) -> bool {
        self.align_to::<int>();
        if ! self.inner.visit_int() { return false; }
        self.bump_past::<int>();
        true
    }

    fn visit_i8(&mut self) -> bool {
        self.align_to::<i8>();
        if ! self.inner.visit_i8() { return false; }
        self.bump_past::<i8>();
        true
    }

    fn visit_i16(&mut self) -> bool {
        self.align_to::<i16>();
        if ! self.inner.visit_i16() { return false; }
        self.bump_past::<i16>();
        true
    }

    fn visit_i32(&mut self) -> bool {
        self.align_to::<i32>();
        if ! self.inner.visit_i32() { return false; }
        self.bump_past::<i32>();
        true
    }

    fn visit_i64(&mut self) -> bool {
        self.align_to::<i64>();
        if ! self.inner.visit_i64() { return false; }
        self.bump_past::<i64>();
        true
    }

    fn visit_uint(&mut self) -> bool {
        self.align_to::<uint>();
        if ! self.inner.visit_uint() { return false; }
        self.bump_past::<uint>();
        true
    }

    fn visit_u8(&mut self) -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_u8() { return false; }
        self.bump_past::<u8>();
        true
    }

    fn visit_u16(&mut self) -> bool {
        self.align_to::<u16>();
        if ! self.inner.visit_u16() { return false; }
        self.bump_past::<u16>();
        true
    }

    fn visit_u32(&mut self) -> bool {
        self.align_to::<u32>();
        if ! self.inner.visit_u32() { return false; }
        self.bump_past::<u32>();
        true
    }

    fn visit_u64(&mut self) -> bool {
        self.align_to::<u64>();
        if ! self.inner.visit_u64() { return false; }
        self.bump_past::<u64>();
        true
    }

    fn visit_f32(&mut self) -> bool {
        self.align_to::<f32>();
        if ! self.inner.visit_f32() { return false; }
        self.bump_past::<f32>();
        true
    }

    fn visit_f64(&mut self) -> bool {
        self.align_to::<f64>();
        if ! self.inner.visit_f64() { return false; }
        self.bump_past::<f64>();
        true
    }

    fn visit_char(&mut self) -> bool {
        self.align_to::<char>();
        if ! self.inner.visit_char() { return false; }
        self.bump_past::<char>();
        true
    }

    fn visit_estr_box(&mut self) -> bool {
        self.align_to::<@str>();
        if ! self.inner.visit_estr_box() { return false; }
        self.bump_past::<@str>();
        true
    }

    fn visit_estr_uniq(&mut self) -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_estr_uniq() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_slice(&mut self) -> bool {
        self.align_to::<&'static str>();
        if ! self.inner.visit_estr_slice() { return false; }
        self.bump_past::<&'static str>();
        true
    }

    fn visit_estr_fixed(&mut self, n: uint,
                        sz: uint,
                        align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_estr_fixed(n, sz, align) { return false; }
        self.bump(sz);
        true
    }

    fn visit_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_box(mtbl, inner) { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_uniq_managed(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq_managed(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_ptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_ptr(mtbl, inner) { return false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_rptr(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&'static u8>();
        if ! self.inner.visit_rptr(mtbl, inner) { return false; }
        self.bump_past::<&'static u8>();
        true
    }

    fn visit_unboxed_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<Vec<()>>();
        // FIXME (#3732): Inner really has to move its own pointers on this one.
        // or else possibly we could have some weird interface wherein we
        // read-off a word from inner's pointers, but the read-word has to
        // always be the same in all sub-pointers? Dubious.
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        true
    }

    fn visit_vec(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_box(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_evec_box(mtbl, inner) { return false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_evec_uniq(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_evec_uniq(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_uniq_managed(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[@u8]>();
        if ! self.inner.visit_evec_uniq_managed(mtbl, inner) { return false; }
        self.bump_past::<~[@u8]>();
        true
    }

    fn visit_evec_slice(&mut self, mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&'static [u8]>();
        if ! self.inner.visit_evec_slice(mtbl, inner) { return false; }
        self.bump_past::<&'static [u8]>();
        true
    }

    fn visit_evec_fixed(&mut self, n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        self.align(align);
        if ! self.inner.visit_evec_fixed(n, sz, align, mtbl, inner) {
            return false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_rec(&mut self, n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_rec_field(&mut self, i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_rec_field(i, name, mtbl, inner) { return false; }
        true
    }

    fn visit_leave_rec(&mut self, n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_class(&mut self, name: &str, named_fields: bool, n_fields: uint, sz: uint,
                         align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(name, named_fields, n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_class_field(&mut self, i: uint, name: &str, named: bool,
                         mtbl: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_class_field(i, name, named, mtbl, inner) {
            return false;
        }
        true
    }

    fn visit_leave_class(&mut self, name: &str, named_fields: bool, n_fields: uint, sz: uint,
                         align: uint) -> bool {
        if ! self.inner.visit_leave_class(name, named_fields, n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_enter_tup(&mut self, n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_tup_field(&mut self, i: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_tup_field(i, inner) { return false; }
        true
    }

    fn visit_leave_tup(&mut self, n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            return false
        }
        true
    }

    fn visit_fn_input(&mut self, i: uint, mode: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_input(i, mode, inner) { return false; }
        true
    }

    fn visit_fn_output(&mut self, retstyle: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_output(retstyle, inner) { return false; }
        true
    }

    fn visit_leave_fn(&mut self, purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn(purity, proto, n_inputs, retstyle) {
            return false;
        }
        true
    }

    fn visit_enter_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        sz: uint, align: uint)
                     -> bool {
        self.align(align);
        if ! self.inner.visit_enter_enum(n_variants, get_disr, sz, align) { return false; }
        true
    }

    fn visit_enter_enum_variant(&mut self, variant: uint,
                                disr_val: Disr,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_enum_variant_field(&mut self, i: uint, offset: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_enum_variant_field(i, offset, inner) { return false; }
        true
    }

    fn visit_leave_enum_variant(&mut self, variant: uint,
                                disr_val: Disr,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_leave_enum(&mut self, n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        sz: uint, align: uint)
                     -> bool {
        if ! self.inner.visit_leave_enum(n_variants, get_disr, sz, align) { return false; }
        true
    }

    fn visit_trait(&mut self, name: &str) -> bool {
        self.align_to::<@TyVisitor>();
        if ! self.inner.visit_trait(name) { return false; }
        self.bump_past::<@TyVisitor>();
        true
    }

    fn visit_param(&mut self, i: uint) -> bool {
        if ! self.inner.visit_param(i) { return false; }
        true
    }

    fn visit_self(&mut self) -> bool {
        self.align_to::<&'static u8>();
        if ! self.inner.visit_self() { return false; }
        self.align_to::<&'static u8>();
        true
    }

    fn visit_type(&mut self) -> bool {
        if ! self.inner.visit_type() { return false; }
        true
    }

    fn visit_opaque_box(&mut self) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_opaque_box() { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_closure_ptr(&mut self, ck: uint) -> bool {
        self.align_to::<(uint,uint)>();
        if ! self.inner.visit_closure_ptr(ck) { return false; }
        self.bump_past::<(uint,uint)>();
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
    pub fn get<T:Clone>(&mut self, f: &fn(T)) {
        unsafe {
            f((*(self.ptr1 as *T)).clone());
        }
    }

    pub fn visit_inner(&mut self, inner: *TyDesc) -> bool {
        unsafe {
            let u = my_visitor(**self);
            let mut v = ptr_visit_adaptor::<my_visitor>(Inner {inner: u});
            visit_tydesc(inner, &mut v as &mut TyVisitor);
            true
        }
    }
}

struct Inner<V> { inner: V }

impl movable_ptr for my_visitor {
    fn move_ptr(&mut self, adjustment: &fn(*c_void) -> *c_void) {
        self.ptr1 = adjustment(self.ptr1);
        self.ptr2 = adjustment(self.ptr2);
    }
}

impl TyVisitor for my_visitor {

    fn visit_bot(&mut self) -> bool { true }
    fn visit_nil(&mut self) -> bool { true }
    fn visit_bool(&mut self) -> bool {
        do self.get::<bool>() |b| {
            self.vals.push(b.to_str());
        };
        true
    }
    fn visit_int(&mut self) -> bool {
        do self.get::<int>() |i| {
            self.vals.push(i.to_str());
        };
        true
    }
    fn visit_i8(&mut self) -> bool { true }
    fn visit_i16(&mut self) -> bool { true }
    fn visit_i32(&mut self) -> bool { true }
    fn visit_i64(&mut self) -> bool { true }

    fn visit_uint(&mut self) -> bool { true }
    fn visit_u8(&mut self) -> bool { true }
    fn visit_u16(&mut self) -> bool { true }
    fn visit_u32(&mut self) -> bool { true }
    fn visit_u64(&mut self) -> bool { true }

    fn visit_f32(&mut self) -> bool { true }
    fn visit_f64(&mut self) -> bool { true }

    fn visit_char(&mut self) -> bool { true }

    fn visit_estr_box(&mut self) -> bool { true }
    fn visit_estr_uniq(&mut self) -> bool { true }
    fn visit_estr_slice(&mut self) -> bool { true }
    fn visit_estr_fixed(&mut self, _n: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq_managed(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_vec(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq_managed(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_slice(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(&mut self, _n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(&mut self, _i: uint, _name: &str,
                       _mtbl: uint, inner: *TyDesc) -> bool {
        error!("rec field!");
        self.visit_inner(inner)
    }
    fn visit_leave_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(&mut self, _name: &str, _named_fields: bool, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(&mut self, _i: uint, _name: &str, _named: bool,
                         _mtbl: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_class(&mut self, _name: &str, _named_fields: bool, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(&mut self, _i: uint, inner: *TyDesc) -> bool {
        error!("tup field!");
        self.visit_inner(inner)
    }
    fn visit_leave_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(&mut self, _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        _sz: uint, _align: uint) -> bool {
        // FIXME (#3732): this needs to rewind between enum variants, or something.
        true
    }
    fn visit_enter_enum_variant(&mut self, _variant: uint,
                                _disr_val: Disr,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(&mut self, _i: uint, _offset: uint, inner: *TyDesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_enum_variant(&mut self, _variant: uint,
                                _disr_val: Disr,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(&mut self, _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> Disr,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(&mut self, _i: uint, _mode: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_fn_output(&mut self, _retstyle: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_leave_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait(&mut self, _name: &str) -> bool { true }
    fn visit_param(&mut self, _i: uint) -> bool { true }
    fn visit_self(&mut self) -> bool { true }
    fn visit_type(&mut self) -> bool { true }
    fn visit_opaque_box(&mut self) -> bool { true }
    fn visit_closure_ptr(&mut self, _ck: uint) -> bool { true }
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
        let mut v = ptr_visit_adaptor(Inner {inner: u});
        let td = get_tydesc_for(r);
        error!("tydesc sz: {}, align: {}",
               (*td).size, (*td).align);
        visit_tydesc(td, &mut v as &mut TyVisitor);

        let r = u.vals.clone();
        for s in r.iter() {
            println!("val: {}", *s);
        }
        error!("{:?}", u.vals.clone());
        assert_eq!(u.vals.clone(),
                   ~[ ~"1", ~"2", ~"3", ~"true", ~"false", ~"5", ~"4", ~"3", ~"12"]);
    }
}
