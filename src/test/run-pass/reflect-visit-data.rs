import libc::c_void;

iface data_cursor {
    fn set_ptr(p: *c_void);
    fn get_ptr() -> *c_void;
}

enum my_visitor = @{
    mut ptr: *c_void,
    mut vals: [str]
};

impl methods for my_visitor {
    fn get<T>(f: fn(T)) {
        unsafe {
            f(*(self.ptr as *T));
        }
    }
}

impl of data_cursor for my_visitor {
    fn set_ptr(p: *c_void) { self.ptr = p; }
    fn get_ptr() -> *c_void { self.ptr }
}

impl of intrinsic::ty_visitor for my_visitor {

    fn visit_bot() -> bool { true }
    fn visit_nil() -> bool { true }
    fn visit_bool() -> bool {
        self.get::<bool>() {|b|
            self.vals += [bool::to_str(b)];
        }
        true
    }
    fn visit_int() -> bool {
        self.get::<int>() {|i|
            self.vals += [int::to_str(i, 10u)];
        }
        true
    }
    fn visit_i8() -> bool { true }
    fn visit_i16() -> bool { true }
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

    fn visit_enter_vec(_mtbl: uint) -> bool { true }
    fn visit_leave_vec(_mtbl: uint) -> bool { true }
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

enum data_visitor<V:intrinsic::ty_visitor data_cursor> = {
    inner: V
};

fn align_to<T>(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

impl dv<V: intrinsic::ty_visitor data_cursor> of
    intrinsic::ty_visitor for data_visitor<V> {

    fn move_ptr(f: fn(*c_void) -> *c_void) {
        self.inner.set_ptr(f(self.inner.get_ptr()));
    }

    fn bump(sz: uint) {
        self.move_ptr() {|p|
            ((p as uint) + sz) as *c_void
        }
    }

    fn align_to<T>() {
        self.move_ptr() {|p|
            align_to::<T>(p as uint,
                          sys::min_align_of::<T>()) as *c_void
        }
    }

    fn bump_past<T>() {
        self.bump(sys::size_of::<T>());
    }

    fn visit_bot() -> bool {
        self.align_to::<bool>();
        self.inner.visit_bot();
        self.bump_past::<bool>();
        true
    }
    fn visit_nil() -> bool { true }
    fn visit_bool() -> bool {
        self.align_to::<bool>();
        self.inner.visit_bool();
        self.bump_past::<bool>();
        true
    }
    fn visit_int() -> bool {
        self.align_to::<int>();
        self.inner.visit_int();
        self.bump_past::<int>();
        true
    }
    fn visit_i8() -> bool { true }
    fn visit_i16() -> bool { true }
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

    fn visit_enter_vec(_mtbl: uint) -> bool { true }
    fn visit_leave_vec(_mtbl: uint) -> bool { true }
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
    let r = (1,2,3,true,false);
    let p = ptr::addr_of(r) as *c_void;
    let u = my_visitor(@{mut ptr: p,
                         mut vals: []});
    let v = data_visitor({inner: u});
    let vv = v as intrinsic::ty_visitor;
    intrinsic::visit_ty::<(int,int,int,bool,bool)>(vv);

    for u.vals.each {|s|
        io::println(#fmt("val: %s", s));
    }
    assert u.vals == ["1", "2", "3", "true", "false"];

 }
