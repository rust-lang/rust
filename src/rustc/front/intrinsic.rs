// NB: this file is #include_str'ed into the compiler, re-parsed
// and injected into each crate the compiler builds. Keep it small.

mod intrinsic {

    import rusti::visit_tydesc;
    export ty_visitor, get_tydesc, visit_tydesc, tydesc;

    // FIXME (#2712): remove this when the interface has settled and the
    // version in sys is no longer present.
    fn get_tydesc<T>() -> *tydesc {
        rusti::get_tydesc::<T>() as *tydesc
    }

    enum tydesc = {
        first_param: **u8,
        size: uint,
        align: uint
        // Remaining fields not listed
    };

    trait ty_visitor {
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
        fn visit_estr_fixed(n: uint, sz: uint, align: uint) -> bool;

        fn visit_box(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_uniq(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_ptr(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_rptr(mtbl: uint, inner: *tydesc) -> bool;

        fn visit_vec(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_unboxed_vec(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_evec_box(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_evec_uniq(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_evec_slice(mtbl: uint, inner: *tydesc) -> bool;
        fn visit_evec_fixed(n: uint, sz: uint, align: uint,
                            mtbl: uint, inner: *tydesc) -> bool;

        fn visit_enter_rec(n_fields: uint,
                           sz: uint, align: uint) -> bool;
        fn visit_rec_field(i: uint, name: &str,
                           mtbl: uint, inner: *tydesc) -> bool;
        fn visit_leave_rec(n_fields: uint,
                           sz: uint, align: uint) -> bool;

        fn visit_enter_class(n_fields: uint,
                             sz: uint, align: uint) -> bool;
        fn visit_class_field(i: uint, name: &str,
                             mtbl: uint, inner: *tydesc) -> bool;
        fn visit_leave_class(n_fields: uint,
                             sz: uint, align: uint) -> bool;

        fn visit_enter_tup(n_fields: uint,
                           sz: uint, align: uint) -> bool;
        fn visit_tup_field(i: uint, inner: *tydesc) -> bool;
        fn visit_leave_tup(n_fields: uint,
                           sz: uint, align: uint) -> bool;

        fn visit_enter_enum(n_variants: uint,
                            sz: uint, align: uint) -> bool;
        fn visit_enter_enum_variant(variant: uint,
                                    disr_val: int,
                                    n_fields: uint,
                                    name: &str) -> bool;
        fn visit_enum_variant_field(i: uint, inner: *tydesc) -> bool;
        fn visit_leave_enum_variant(variant: uint,
                                    disr_val: int,
                                    n_fields: uint,
                                    name: &str) -> bool;
        fn visit_leave_enum(n_variants: uint,
                            sz: uint, align: uint) -> bool;

        fn visit_enter_fn(purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;
        fn visit_fn_input(i: uint, mode: uint, inner: *tydesc) -> bool;
        fn visit_fn_output(retstyle: uint, inner: *tydesc) -> bool;
        fn visit_leave_fn(purity: uint, proto: uint,
                          n_inputs: uint, retstyle: uint) -> bool;

        fn visit_trait() -> bool;
        fn visit_var() -> bool;
        fn visit_var_integral() -> bool;
        fn visit_param(i: uint) -> bool;
        fn visit_self() -> bool;
        fn visit_type() -> bool;
        fn visit_opaque_box() -> bool;
        fn visit_constr(inner: *tydesc) -> bool;
        fn visit_closure_ptr(ck: uint) -> bool;
    }

    #[abi = "rust-intrinsic"]
    extern mod rusti {
        fn get_tydesc<T>() -> *();
        fn visit_tydesc(td: *tydesc, &&tv: ty_visitor);
    }
}
