// xfail-test
//
// This doesn't work quite yet in check-fast mode. Not sure why. Crashes.

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

    fn visit_vec_of(_mutbl: uint) -> bool { true }
    fn visit_box_of(_mutbl: uint) -> bool { true }
    fn visit_uniq_of(_mutbl: uint) -> bool { true }
    fn visit_ptr_of(_mutbl: uint) -> bool { true }
    fn visit_rptr_of(_mutbl: uint) -> bool { true }
    fn visit_rec_of(_n_fields: uint) -> bool { true }
    fn visit_rec_field(_name: str/&, _mutbl: uint) -> bool { true }
    fn visit_tup_of(_n_fields: uint) -> bool { true }
    fn visit_tup_field(_mutbl: uint) -> bool { true }
    fn visit_enum_of(_n_variants: uint) -> bool { true }
    fn visit_enum_variant(_name: str/&) -> bool { true }
}

fn main() {
    let v = my_visitor(@{mut types: []});
    let vv = v as intrinsic::ty_visitor;

    intrinsic::visit_ty::<bool>(vv);
    intrinsic::visit_ty::<int>(vv);
    intrinsic::visit_ty::<i8>(vv);
    intrinsic::visit_ty::<i16>(vv);

    for v.types.each {|s|
        io::println(#fmt("type: %s", s));
    }
    assert v.types == ["bool", "int", "i8", "i16"];
}
