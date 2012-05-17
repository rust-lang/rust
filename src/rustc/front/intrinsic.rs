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

        // FIXME: possibly pair these as enter/leave calls
        // not just enter with implicit number of subsequent
        // calls. (#2402)
        fn visit_vec_of(mutbl: uint) -> bool;
        fn visit_box_of(mutbl: uint) -> bool;
        fn visit_uniq_of(mutbl: uint) -> bool;
        fn visit_ptr_of(mutbl: uint) -> bool;
        fn visit_rptr_of(mutbl: uint) -> bool;
        fn visit_rec_of(n_fields: uint) -> bool;
        fn visit_rec_field(name: str/&, mutbl: uint) -> bool;
        fn visit_tup_of(n_fields: uint) -> bool;
        fn visit_tup_field(mutbl: uint) -> bool;
        fn visit_enum_of(n_variants: uint) -> bool;
        fn visit_enum_variant(name: str/&) -> bool;
    }

    #[abi = "rust-intrinsic"]
    native mod rusti {
        fn visit_ty<T>(&&tv: ty_visitor);
    }
}
