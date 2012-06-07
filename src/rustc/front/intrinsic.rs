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
    }

    #[abi = "rust-intrinsic"]
    native mod rusti {
        fn visit_ty<T>(&&tv: ty_visitor);
    }
}
