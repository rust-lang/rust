// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// NB: this file is include_str!'ed into the compiler, re-parsed
// and injected into each crate the compiler builds. Keep it small.

pub mod intrinsic {
    pub use intrinsic::rusti::visit_tydesc;

    // FIXME (#3727): remove this when the interface has settled and the
    // version in sys is no longer present.
    pub fn get_tydesc<T>() -> *TyDesc {
        unsafe {
            rusti::get_tydesc::<T>() as *TyDesc
        }
    }

    pub struct TyDesc {
        size: uint,
        align: uint
        // Remaining fields not listed
    }

    pub enum Opaque { }

    pub trait TyVisitor {
        fn visit_bot(&self) -> bool;
        fn visit_nil(&self) -> bool;
        fn visit_bool(&self) -> bool;

        fn visit_int(&self) -> bool;
        fn visit_i8(&self) -> bool;
        fn visit_i16(&self) -> bool;
        fn visit_i32(&self) -> bool;
        fn visit_i64(&self) -> bool;

        fn visit_uint(&self) -> bool;
        fn visit_u8(&self) -> bool;
        fn visit_u16(&self) -> bool;
        fn visit_u32(&self) -> bool;
        fn visit_u64(&self) -> bool;

        fn visit_float(&self) -> bool;
        fn visit_f32(&self) -> bool;
        fn visit_f64(&self) -> bool;

        fn visit_char(&self) -> bool;
        fn visit_str(&self) -> bool;

        fn visit_estr_box(&self) -> bool;
        fn visit_estr_uniq(&self) -> bool;
        fn visit_estr_slice(&self) -> bool;
        fn visit_estr_fixed(&self, n: uint, sz: uint, align: uint) -> bool;

        fn visit_box(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_ptr(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_rptr(&self, mtbl: uint, inner: *TyDesc) -> bool;

        fn visit_vec(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_unboxed_vec(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_evec_box(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_evec_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_evec_slice(&self, mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_evec_fixed(&self, n: uint, sz: uint, align: uint,
                            mtbl: uint, inner: *TyDesc) -> bool;

        fn visit_enter_rec(&self, n_fields: uint,
                           sz: uint, align: uint) -> bool;
        fn visit_rec_field(&self, i: uint, name: &str,
                           mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_leave_rec(&self, n_fields: uint,
                           sz: uint, align: uint) -> bool;

        fn visit_enter_class(&self, n_fields: uint,
                             sz: uint, align: uint) -> bool;
        fn visit_class_field(&self, i: uint, name: &str,
                             mtbl: uint, inner: *TyDesc) -> bool;
        fn visit_leave_class(&self, n_fields: uint,
                             sz: uint, align: uint) -> bool;

        fn visit_enter_tup(&self, n_fields: uint,
                           sz: uint, align: uint) -> bool;
        fn visit_tup_field(&self, i: uint, inner: *TyDesc) -> bool;
        fn visit_leave_tup(&self, n_fields: uint,
                           sz: uint, align: uint) -> bool;

        fn visit_enter_enum(&self, n_variants: uint,
                            get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                            sz: uint, align: uint) -> bool;
        fn visit_enter_enum_variant(&self, variant: uint,
                                    disr_val: int,
                                    n_fields: uint,
                                    name: &str) -> bool;
        fn visit_enum_variant_field(&self, i: uint, offset: uint, inner: *TyDesc) -> bool;
        fn visit_leave_enum_variant(&self, variant: uint,
                                    disr_val: int,
                                    n_fields: uint,
                                    name: &str) -> bool;
        fn visit_leave_enum(&self, n_variants: uint,
                            get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                            sz: uint, align: uint) -> bool;

        fn visit_enter_fn(&self, purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;
        fn visit_fn_input(&self, i: uint, mode: uint, inner: *TyDesc) -> bool;
        fn visit_fn_output(&self, retstyle: uint, inner: *TyDesc) -> bool;
        fn visit_leave_fn(&self, purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;

        fn visit_trait(&self) -> bool;
        fn visit_var(&self) -> bool;
        fn visit_var_integral(&self) -> bool;
        fn visit_param(&self, i: uint) -> bool;
        fn visit_self(&self) -> bool;
        fn visit_type(&self) -> bool;
        fn visit_opaque_box(&self) -> bool;
        fn visit_constr(&self, inner: *TyDesc) -> bool;
        fn visit_closure_ptr(&self, ck: uint) -> bool;
    }

    pub mod rusti {
        use super::{TyDesc, TyVisitor};

        #[abi = "rust-intrinsic"]
        pub extern "rust-intrinsic" {
            pub fn get_tydesc<T>() -> *();
            pub fn visit_tydesc(++td: *TyDesc, ++tv: @TyVisitor);
        }
    }
}
