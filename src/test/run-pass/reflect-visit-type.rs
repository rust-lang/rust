// xfail-test
//
// This works on stage2 currently. Once we have a snapshot
// and some fiddling with inject_intrinsic (and possibly another
// snapshot after _that_) it can be un-xfailed and changed
// to use the intrinsic:: interface and native module.
//

iface ty_visitor {
    fn visit_bot();
    fn visit_nil();
    fn visit_bool();

    fn visit_int();
    fn visit_i8();
    fn visit_i16();
    fn visit_i32();
    fn visit_i64();

    fn visit_uint();
    fn visit_u8();
    fn visit_u16();
    fn visit_u32();
    fn visit_u64();

    fn visit_float();
    fn visit_f32();
    fn visit_f64();

    fn visit_char();
    fn visit_str();

    fn visit_vec(cells_mut: bool,
                 visit_cell: fn(uint, self));

    fn visit_box(inner_mut: bool,
                 visit_inner: fn(self));

    fn visit_uniq(inner_mut: bool,
                  visit_inner: fn(self));

    fn visit_ptr(inner_mut: bool,
                 visit_inner: fn(self));

    fn visit_rptr(inner_mut: bool,
                  visit_inner: fn(self));

    fn visit_rec(n_fields: uint,
                 field_name: fn(uint) -> str/&,
                 field_mut: fn(uint) -> bool,
                 visit_field: fn(uint, self));
    fn visit_tup(n_fields: uint,
                 visit_field: fn(uint, self));
    fn visit_enum(n_variants: uint,
                  variant: uint,
                  variant_name: fn(uint) -> str/&,
                  visit_variant: fn(uint, self));
}

enum my_visitor = { mut types: [str] };

impl of ty_visitor for my_visitor {
    fn visit_bot() { self.types += ["bot"] }
    fn visit_nil() { self.types += ["nil"] }
    fn visit_bool() { self.types += ["bool"] }

    fn visit_int() { self.types += ["int"] }
    fn visit_i8() { self.types += ["i8"] }
    fn visit_i16() { self.types += ["i16"] }
    fn visit_i32() { }
    fn visit_i64() { }

    fn visit_uint() { }
    fn visit_u8() { }
    fn visit_u16() { }
    fn visit_u32() { }
    fn visit_u64() { }

    fn visit_float() { }
    fn visit_f32() { }
    fn visit_f64() { }

    fn visit_char() { }
    fn visit_str() { }

    fn visit_vec(_cells_mut: bool,
                 _visit_cell: fn(uint, my_visitor)) { }

    fn visit_box(_inner_mut: bool,
                 _visit_inner: fn(my_visitor)) { }

    fn visit_uniq(_inner_mut: bool,
                  _visit_inner: fn(my_visitor)) { }

    fn visit_ptr(_inner_mut: bool,
                 _visit_inner: fn(my_visitor)) { }

    fn visit_rptr(_inner_mut: bool,
                  _visit_inner: fn(my_visitor)) { }

    fn visit_rec(_n_fields: uint,
                 _field_name: fn(uint) -> str/&,
                 _field_mut: fn(uint) -> bool,
                 _visit_field: fn(uint, my_visitor)) { }
    fn visit_tup(_n_fields: uint,
                 _visit_field: fn(uint, my_visitor)) { }
    fn visit_enum(_n_variants: uint,
                  _variant: uint,
                  _variant_name: fn(uint) -> str/&,
                  _visit_variant: fn(uint, my_visitor)) { }
}

#[abi = "rust-intrinsic"]
native mod rusti {
    fn visit_ty<T,V:ty_visitor>(tv: V);
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
}
