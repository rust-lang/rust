enum my_visitor = @{ mut types: [str] };

impl of intrinsic::ty_visitor for my_visitor {
    fn visit_bot() -> bool {
        self.types += ["bot"];
        #error("visited bot type");
        true
    }
    fn visit_nil() -> bool {
        self.types += ["nil"];
        #error("visited nil type");
        true
    }
    fn visit_bool() -> bool {
        self.types += ["bool"];
        #error("visited bool type");
        true
    }
    fn visit_int() -> bool {
        self.types += ["int"];
        #error("visited int type");
        true
    }
    fn visit_i8() -> bool {
        self.types += ["i8"];
        #error("visited i8 type");
        true
    }
    fn visit_i16() -> bool {
        self.types += ["i16"];
        #error("visited i16 type");
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
    fn visit_estr_fixed(_sz: uint) -> bool { true }

    fn visit_enter_box(_mtbl: uint) -> bool { true }
    fn visit_leave_box(_mtbl: uint) -> bool { true }
    fn visit_enter_uniq(_mtbl: uint) -> bool { true }
    fn visit_leave_uniq(_mtbl: uint) -> bool { true }
    fn visit_enter_ptr(_mtbl: uint) -> bool { true }
    fn visit_leave_ptr(_mtbl: uint) -> bool { true }
    fn visit_enter_rptr(_mtbl: uint) -> bool { true }
    fn visit_leave_rptr(_mtbl: uint) -> bool { true }

    fn visit_enter_vec(_mtbl: uint) -> bool {
        self.types += ["["];
        #error("visited enter-vec");
        true
    }
    fn visit_leave_vec(_mtbl: uint) -> bool {
        self.types += ["]"];
        #error("visited leave-vec");
        true
    }
    fn visit_enter_evec_box(_mtbl: uint) -> bool { true }
    fn visit_leave_evec_box(_mtbl: uint) -> bool { true }
    fn visit_enter_evec_uniq(_mtbl: uint) -> bool { true }
    fn visit_leave_evec_uniq(_mtbl: uint) -> bool { true }
    fn visit_enter_evec_slice(_mtbl: uint) -> bool { true }
    fn visit_leave_evec_slice(_mtbl: uint) -> bool { true }
    fn visit_enter_evec_fixed(_mtbl: uint, _sz: uint) -> bool { true }
    fn visit_leave_evec_fixed(_mtbl: uint, _sz: uint) -> bool { true }

    fn visit_enter_rec(_n_fields: uint) -> bool { true }
    fn visit_enter_rec_field(_mtbl: uint, _i: uint,
                             _name: str/&) -> bool { true }
    fn visit_leave_rec_field(_mtbl: uint, _i: uint,
                             _name: str/&) -> bool { true }
    fn visit_leave_rec(_n_fields: uint) -> bool { true }

    fn visit_enter_class(_n_fields: uint) -> bool { true }
    fn visit_enter_class_field(_mtbl: uint, _i: uint,
                               _name: str/&) -> bool { true }
    fn visit_leave_class_field(_mtbl: uint, _i: uint,
                               _name: str/&) -> bool { true }
    fn visit_leave_class(_n_fields: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint) -> bool { true }
    fn visit_enter_tup_field(_i: uint) -> bool { true }
    fn visit_leave_tup_field(_i: uint) -> bool { true }
    fn visit_leave_tup(_n_fields: uint) -> bool { true }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_enter_fn_input(_i: uint, _mode: uint) -> bool { true }
    fn visit_leave_fn_input(_i: uint, _mode: uint) -> bool { true }
    fn visit_enter_fn_output(_retstyle: uint) -> bool { true }
    fn visit_leave_fn_output(_retstyle: uint) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }

    fn visit_enter_enum(_n_variants: uint) -> bool { true }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: str/&) -> bool { true }
    fn visit_enter_enum_variant_field(_i: uint) -> bool { true }
    fn visit_leave_enum_variant_field(_i: uint) -> bool { true }
    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: str/&) -> bool { true }
    fn visit_leave_enum(_n_variants: uint) -> bool { true }

    fn visit_iface() -> bool { true }
    fn visit_enter_res() -> bool { true }
    fn visit_leave_res() -> bool { true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }
    fn visit_opaque_box() -> bool { true }
    fn visit_enter_constr() -> bool { true }
    fn visit_leave_constr() -> bool { true }
    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

fn main() {
    let v = my_visitor(@{mut types: []});
    let vv = v as intrinsic::ty_visitor;

    intrinsic::visit_ty::<bool>(vv);
    intrinsic::visit_ty::<int>(vv);
    intrinsic::visit_ty::<i8>(vv);
    intrinsic::visit_ty::<i16>(vv);
    intrinsic::visit_ty::<[int]>(vv);

    for (copy v.types).each {|s|
        io::println(#fmt("type: %s", s));
    }
    assert v.types == ["bool", "int", "i8", "i16",
                       "[", "int", "]"];
}
