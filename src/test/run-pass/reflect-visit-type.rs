// xfail-test
//
// This works on stage2 currently. Once we have a snapshot
// and some fiddling with inject_intrinsic (and possibly another
// snapshot after _that_) it can be un-xfailed and changed
// to use the intrinsic:: interface and native module.
//

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
    // calls.
    fn visit_vec_of(is_mut: bool) -> bool;
    fn visit_box_of(is_mut: bool) -> bool;
    fn visit_uniq_of(is_mut: bool) -> bool;
    fn visit_ptr_of(is_mut: bool) -> bool;
    fn visit_rptr_of(is_mut: bool) -> bool;
    fn visit_rec_of(n_fields: uint) -> bool;
    fn visit_rec_field(name: str/&, is_mut: bool) -> bool;
    fn visit_tup_of(n_fields: uint) -> bool;
    fn visit_tup_field(is_mut: bool) -> bool;
    fn visit_enum_of(n_variants: uint) -> bool;
    fn visit_enum_variant(name: str/&) -> bool;
}

enum my_visitor = { mut types: [str] };

impl of ty_visitor for my_visitor {
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

    fn visit_vec_of(_is_mut: bool) -> bool { true }
    fn visit_box_of(_is_mut: bool) -> bool { true }
    fn visit_uniq_of(_is_mut: bool) -> bool { true }
    fn visit_ptr_of(_is_mut: bool) -> bool { true }
    fn visit_rptr_of(_is_mut: bool) -> bool { true }
    fn visit_rec_of(_n_fields: uint) -> bool { true }
    fn visit_rec_field(_name: str/&, _is_mut: bool) -> bool { true }
    fn visit_tup_of(_n_fields: uint) -> bool { true }
    fn visit_tup_field(_is_mut: bool) -> bool { true }
    fn visit_enum_of(_n_variants: uint) -> bool { true }
    fn visit_enum_variant(_name: str/&) -> bool { true }
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn visit_ty<T,V:ty_visitor>(tv: V);
}

fn via_iface(v: ty_visitor) {
    rusti::visit_ty::<bool,ty_visitor>(v);
    rusti::visit_ty::<int,ty_visitor>(v);
    rusti::visit_ty::<i8,ty_visitor>(v);
    rusti::visit_ty::<i16,ty_visitor>(v);
}

fn main() {
    let v = my_visitor({mut types: []});

    rusti::visit_ty::<bool,my_visitor>(v);
    rusti::visit_ty::<int,my_visitor>(v);
    rusti::visit_ty::<i8,my_visitor>(v);
    rusti::visit_ty::<i16,my_visitor>(v);

    for v.types.each {|s|
        io::println(#fmt("type: %s", s));
    }
    assert v.types == ["bool", "int", "i8", "i16"];

    via_iface(v as ty_visitor);
}
