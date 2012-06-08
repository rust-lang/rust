// NB: this file is #include_str'ed into the compiler, re-parsed
// and injected into each crate the compiler builds. Keep it small.

mod intrinsic {

    import rusti::visit_ty;
    export ty_visitor, visit_ty;

    iface ty_visitor {
        fn visit_bot() -> bool;
        fn visit_nil() -> bool;
        fn visit_bool() -> bool;

        fn visit_int() -> bool;
        fn visit_i8() -> bool;
        fn visit_i16() -> bool;
        fn visit_i32() -> bool;
        fn visit_i64() -> bool;

        fn visit_uint() -> bool;
        fn visit_u8() -> bool;
        fn visit_u16() -> bool;
        fn visit_u32() -> bool;
        fn visit_u64() -> bool;

        fn visit_float() -> bool;
        fn visit_f32() -> bool;
        fn visit_f64() -> bool;

        fn visit_char() -> bool;
        fn visit_str() -> bool;

        fn visit_estr_box() -> bool;
        fn visit_estr_uniq() -> bool;
        fn visit_estr_slice() -> bool;
        fn visit_estr_fixed(sz: uint) -> bool;

        fn visit_enter_box(mtbl: uint) -> bool;
        fn visit_leave_box(mtbl: uint) -> bool;
        fn visit_enter_uniq(mtbl: uint) -> bool;
        fn visit_leave_uniq(mtbl: uint) -> bool;
        fn visit_enter_ptr(mtbl: uint) -> bool;
        fn visit_leave_ptr(mtbl: uint) -> bool;
        fn visit_enter_rptr(mtbl: uint) -> bool;
        fn visit_leave_rptr(mtbl: uint) -> bool;

        fn visit_enter_vec(mtbl: uint) -> bool;
        fn visit_leave_vec(mtbl: uint) -> bool;
        fn visit_enter_evec_box(mtbl: uint) -> bool;
        fn visit_leave_evec_box(mtbl: uint) -> bool;
        fn visit_enter_evec_uniq(mtbl: uint) -> bool;
        fn visit_leave_evec_uniq(mtbl: uint) -> bool;
        fn visit_enter_evec_slice(mtbl: uint) -> bool;
        fn visit_leave_evec_slice(mtbl: uint) -> bool;
        fn visit_enter_evec_fixed(mtbl: uint, sz: uint) -> bool;
        fn visit_leave_evec_fixed(mtbl: uint, sz: uint) -> bool;

        fn visit_enter_rec(n_fields: uint) -> bool;
        fn visit_enter_rec_field(mtbl: uint, i: uint
                                 /*, name: str/& */) -> bool;
        fn visit_leave_rec_field(mtbl: uint, i: uint
                                 /*, name: str/& */) -> bool;
        fn visit_leave_rec(n_fields: uint) -> bool;

        fn visit_enter_class(n_fields: uint) -> bool;
        fn visit_enter_class_field(mtbl: uint, i: uint
                                   /*, name: str/& */) -> bool;
        fn visit_leave_class_field(mtbl: uint, i: uint
                                   /*, name: str/& */) -> bool;
        fn visit_leave_class(n_fields: uint) -> bool;

        fn visit_enter_tup(n_fields: uint) -> bool;
        fn visit_enter_tup_field(i: uint) -> bool;
        fn visit_leave_tup_field(i: uint) -> bool;
        fn visit_leave_tup(n_fields: uint) -> bool;

        fn visit_enter_enum(n_variants: uint) -> bool;
        fn visit_enter_enum_variant(variant: uint,
                                    disr_val: int,
                                    n_fields: uint) -> bool;
        fn visit_enter_enum_variant_field(i: uint) -> bool;
        fn visit_leave_enum_variant_field(i: uint) -> bool;
        fn visit_leave_enum_variant(variant: uint,
                                    disr_val: int,
                                    n_fields: uint) -> bool;
        fn visit_leave_enum(n_variants: uint) -> bool;

        fn visit_enter_fn(purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;
        fn visit_enter_fn_input(i: uint, mode: uint) -> bool;
        fn visit_leave_fn_input(i: uint, mode: uint) -> bool;
        fn visit_enter_fn_output(retstyle: uint) -> bool;
        fn visit_leave_fn_output(retstyle: uint) -> bool;
        fn visit_leave_fn(purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;

        fn visit_iface() -> bool;
        fn visit_enter_res() -> bool;
        fn visit_leave_res() -> bool;
        fn visit_var() -> bool;
        fn visit_var_integral() -> bool;
        fn visit_param(i: uint) -> bool;
        fn visit_self() -> bool;
        fn visit_type() -> bool;
        fn visit_opaque_box() -> bool;
        fn visit_enter_constr() -> bool;
        fn visit_leave_constr() -> bool;
        fn visit_closure_ptr(ck: uint) -> bool;
    }

    #[abi = "rust-intrinsic"]
    native mod rusti {
        fn visit_ty<T>(&&tv: ty_visitor);
    }
}
