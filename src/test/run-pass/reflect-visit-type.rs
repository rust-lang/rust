// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]

use std::intrinsics::{TyDesc, get_tydesc, visit_tydesc, TyVisitor, Disr, Opaque};

struct MyVisitor {
    types: Vec<String> ,
}

impl TyVisitor for MyVisitor {
    fn visit_bot(&mut self) -> bool {
        self.types.push("bot".to_string());
        println!("visited bot type");
        true
    }
    fn visit_nil(&mut self) -> bool {
        self.types.push("nil".to_string());
        println!("visited nil type");
        true
    }
    fn visit_bool(&mut self) -> bool {
        self.types.push("bool".to_string());
        println!("visited bool type");
        true
    }
    fn visit_int(&mut self) -> bool {
        self.types.push("int".to_string());
        println!("visited int type");
        true
    }
    fn visit_i8(&mut self) -> bool {
        self.types.push("i8".to_string());
        println!("visited i8 type");
        true
    }
    fn visit_i16(&mut self) -> bool {
        self.types.push("i16".to_string());
        println!("visited i16 type");
        true
    }
    fn visit_i32(&mut self) -> bool { true }
    fn visit_i64(&mut self) -> bool { true }

    fn visit_uint(&mut self) -> bool { true }
    fn visit_u8(&mut self) -> bool { true }
    fn visit_u16(&mut self) -> bool { true }
    fn visit_u32(&mut self) -> bool { true }
    fn visit_u64(&mut self) -> bool { true }

    fn visit_f32(&mut self) -> bool { true }
    fn visit_f64(&mut self) -> bool { true }
    fn visit_f128(&mut self) -> bool { true }

    fn visit_char(&mut self) -> bool { true }

    fn visit_estr_box(&mut self) -> bool { true }
    fn visit_estr_uniq(&mut self) -> bool { true }
    fn visit_estr_slice(&mut self) -> bool { true }
    fn visit_estr_fixed(&mut self,
                        _sz: uint, _sz2: uint,
                        _align: uint) -> bool { true }

    fn visit_box(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_evec_box(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(&mut self, _mtbl: uint, inner: *TyDesc) -> bool {
        self.types.push("[".to_string());
        unsafe { visit_tydesc(inner, &mut *self as &mut TyVisitor); }
        self.types.push("]".to_string());
        true
    }
    fn visit_evec_slice(&mut self, _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(&mut self, _n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(&mut self, _i: uint, _name: &str,
                       _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(&mut self, _name: &str, _named_fields: bool, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(&mut self, _i: uint, _name: &str, _named: bool,
                         _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_class(&mut self, _name: &str, _named_fields: bool, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(&mut self, _i: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(&mut self, _n_variants: uint,
                        _get_disr: unsafe extern fn(ptr: *Opaque) -> Disr,
                        _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_enum_variant(&mut self,
                                _variant: uint,
                                _disr_val: Disr,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(&mut self, _i: uint, _offset: uint, _inner: *TyDesc)
        -> bool { true }
    fn visit_leave_enum_variant(&mut self,
                                _variant: uint,
                                _disr_val: Disr,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(&mut self,
                        _n_variants: uint,
                        _get_disr: unsafe extern fn(ptr: *Opaque) -> Disr,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(&mut self, _i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(&mut self, _retstyle: uint, _variadic: bool, _inner: *TyDesc)
        -> bool { true }
    fn visit_leave_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait(&mut self, _name: &str) -> bool { true }
    fn visit_param(&mut self, _i: uint) -> bool { true }
    fn visit_self(&mut self) -> bool { true }
}

fn visit_ty<T>(v: &mut MyVisitor) {
    unsafe { visit_tydesc(get_tydesc::<T>(), v as &mut TyVisitor) }
}

pub fn main() {
    let mut v = MyVisitor {types: Vec::new()};

    visit_ty::<bool>(&mut v);
    visit_ty::<int>(&mut v);
    visit_ty::<i8>(&mut v);
    visit_ty::<i16>(&mut v);

    for s in v.types.iter() {
        println!("type: {}", (*s).clone());
    }

    let vec_types: Vec<String> = v.types.clone().move_iter().collect();
    assert_eq!(vec_types, vec!("bool".to_string(), "int".to_string(),
                               "i8".to_string(), "i16".to_string()));
}
