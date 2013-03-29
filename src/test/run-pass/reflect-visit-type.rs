// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
use intrinsic::{TyDesc, get_tydesc, visit_tydesc, TyVisitor};
struct my_visitor(@mut { types: ~[str] });

impl TyVisitor for my_visitor {
    fn visit_bot() -> bool {
        self.types += ~["bot"];
        error!("visited bot type");
        true
    }
    fn visit_nil() -> bool {
        self.types += ~["nil"];
        error!("visited nil type");
        true
    }
    fn visit_bool() -> bool {
        self.types += ~["bool"];
        error!("visited bool type");
        true
    }
    fn visit_int() -> bool {
        self.types += ~["int"];
        error!("visited int type");
        true
    }
    fn visit_i8() -> bool {
        self.types += ~["i8"];
        error!("visited i8 type");
        true
    }
    fn visit_i16() -> bool {
        self.types += ~["i16"];
        error!("visited i16 type");
        true
    }
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
    fn visit_estr_fixed(_sz: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_uniq(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_ptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_vec(_mtbl: uint, inner: *TyDesc) -> bool {
        self.types += ~["["];
        visit_tydesc(inner, my_visitor(*self) as TyVisitor);
        self.types += ~["]"];
        true
    }
    fn visit_unboxed_vec(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_uniq(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_slice(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_fixed(_n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool { true }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(_i: uint, _name: &str,
                       _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(_i: uint, _name: &str,
                         _mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(_i: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(_i: uint, _inner: *TyDesc) -> bool { true }
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

fn visit_ty<T>(v: TyVisitor) {
    visit_tydesc(get_tydesc::<T>(), v);
}

pub fn main() {
    let v = my_visitor(@mut {types: ~[]});
    let vv = v as TyVisitor;

    visit_ty::<bool>(vv);
    visit_ty::<int>(vv);
    visit_ty::<i8>(vv);
    visit_ty::<i16>(vv);
    visit_ty::<~[int]>(vv);

    for (v.types.clone()).each {|s|
        io::println(fmt!("type: %s", s));
    }
    assert!(v.types == ["bool", "int", "i8", "i16",
                       "[", "int", "]"]);
}
