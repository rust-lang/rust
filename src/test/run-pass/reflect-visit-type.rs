// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::unstable::intrinsics::{TyDesc, get_tydesc, visit_tydesc, TyVisitor, Opaque};

struct MyVisitor {
    types: @mut ~[~str],
}

impl TyVisitor for MyVisitor {
    fn visit_bot(&self) -> bool {
        self.types.push(~"bot");
        error!("visited bot type");
        true
    }
    fn visit_nil(&self) -> bool {
        self.types.push(~"nil");
        error!("visited nil type");
        true
    }
    fn visit_bool(&self) -> bool {
        self.types.push(~"bool");
        error!("visited bool type");
        true
    }
    fn visit_int(&self) -> bool {
        self.types.push(~"int");
        error!("visited int type");
        true
    }
    fn visit_i8(&self) -> bool {
        self.types.push(~"i8");
        error!("visited i8 type");
        true
    }
    fn visit_i16(&self) -> bool {
        self.types.push(~"i16");
        error!("visited i16 type");
        true
    }
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
    fn visit_estr_fixed(&self,
                        _sz: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq_managed(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_vec(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(&self, _mtbl: uint, inner: *TyDesc) -> bool {
        self.types.push(~"[");
        unsafe {
            visit_tydesc(inner, (@*self) as @TyVisitor);
        }
        self.types.push(~"]");
        true
    }
    fn visit_evec_uniq_managed(&self, _mtbl: uint, inner: *TyDesc) -> bool {
        self.types.push(~"[");
        unsafe {
            visit_tydesc(inner, (@*self) as @TyVisitor);
        }
        self.types.push(~"]");
        true
    }
    fn visit_evec_slice(&self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(&self, _n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(&self, _i: uint, _name: &str,
                       _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(&self, _i: uint, _name: &str,
                         _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(&self, _i: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(&self, _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_enum_variant(&self,
                                _variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(&self, _i: uint, _offset: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_enum_variant(&self,
                                _variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(&self,
                        _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(&self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(&self, _i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(&self, _retstyle: uint, _inner: *TyDesc) -> bool { true }
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

fn visit_ty<T>(v: @TyVisitor) {
    unsafe {
        visit_tydesc(get_tydesc::<T>(), v);
    }
}

pub fn main() {
    let v = @MyVisitor {types: @mut ~[]};
    let vv = v as @TyVisitor;

    visit_ty::<bool>(vv);
    visit_ty::<int>(vv);
    visit_ty::<i8>(vv);
    visit_ty::<i16>(vv);
    visit_ty::<~[int]>(vv);

    for v.types.iter().advance |s| {
        println(fmt!("type: %s", (*s).clone()));
    }
    assert_eq!((*v.types).clone(), ~[~"bool", ~"int", ~"i8", ~"i16", ~"[", ~"int", ~"]"]);
}
