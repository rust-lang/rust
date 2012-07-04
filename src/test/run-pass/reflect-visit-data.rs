// FIXME: un-xfail after snapshot
// xfail-test

import intrinsic::ty_visitor;
import libc::c_void;

/// High-level interfaces to `intrinsic::visit_ty` reflection system.

/// Iface for visitor that wishes to reflect on data.
iface movable_ptr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void);
}

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

enum ptr_visit_adaptor<V: ty_visitor movable_ptr> = {
    inner: V
};
impl ptr_visitor<V: ty_visitor movable_ptr>
    of ty_visitor for ptr_visit_adaptor<V> {

    #[inline(always)]
    fn bump(sz: uint) {
        self.inner.move_ptr() {|p|
            ((p as uint) + sz) as *c_void
        }
    }

    #[inline(always)]
    fn align(a: uint) {
        self.inner.move_ptr() {|p|
            align(p as uint, a) as *c_void
        }
    }

    #[inline(always)]
    fn align_to<T>() {
        self.align(sys::min_align_of::<T>());
    }

    #[inline(always)]
    fn bump_past<T>() {
        self.bump(sys::size_of::<T>());
    }

    fn visit_bot() -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_bot() { ret false; }
        self.bump_past::<()>();
        true
    }

    fn visit_nil() -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_nil() { ret false; }
        self.bump_past::<()>();
        true
    }

    fn visit_bool() -> bool {
        self.align_to::<bool>();
        if ! self.inner.visit_bool() { ret false; }
        self.bump_past::<bool>();
        true
    }

    fn visit_int() -> bool {
        self.align_to::<int>();
        if ! self.inner.visit_int() { ret false; }
        self.bump_past::<int>();
        true
    }

    fn visit_i8() -> bool {
        self.align_to::<i8>();
        if ! self.inner.visit_i8() { ret false; }
        self.bump_past::<i8>();
        true
    }

    fn visit_i16() -> bool {
        self.align_to::<i16>();
        if ! self.inner.visit_i16() { ret false; }
        self.bump_past::<i16>();
        true
    }

    fn visit_i32() -> bool {
        self.align_to::<i32>();
        if ! self.inner.visit_i32() { ret false; }
        self.bump_past::<i32>();
        true
    }

    fn visit_i64() -> bool {
        self.align_to::<i64>();
        if ! self.inner.visit_i64() { ret false; }
        self.bump_past::<i64>();
        true
    }

    fn visit_uint() -> bool {
        self.align_to::<uint>();
        if ! self.inner.visit_uint() { ret false; }
        self.bump_past::<uint>();
        true
    }

    fn visit_u8() -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_u8() { ret false; }
        self.bump_past::<u8>();
        true
    }

    fn visit_u16() -> bool {
        self.align_to::<u16>();
        if ! self.inner.visit_u16() { ret false; }
        self.bump_past::<u16>();
        true
    }

    fn visit_u32() -> bool {
        self.align_to::<u32>();
        if ! self.inner.visit_u32() { ret false; }
        self.bump_past::<u32>();
        true
    }

    fn visit_u64() -> bool {
        self.align_to::<u64>();
        if ! self.inner.visit_u64() { ret false; }
        self.bump_past::<u64>();
        true
    }

    fn visit_float() -> bool {
        self.align_to::<float>();
        if ! self.inner.visit_float() { ret false; }
        self.bump_past::<float>();
        true
    }

    fn visit_f32() -> bool {
        self.align_to::<f32>();
        if ! self.inner.visit_f32() { ret false; }
        self.bump_past::<f32>();
        true
    }

    fn visit_f64() -> bool {
        self.align_to::<f64>();
        if ! self.inner.visit_f64() { ret false; }
        self.bump_past::<f64>();
        true
    }

    fn visit_char() -> bool {
        self.align_to::<char>();
        if ! self.inner.visit_char() { ret false; }
        self.bump_past::<char>();
        true
    }

    fn visit_str() -> bool {
        self.align_to::<str>();
        if ! self.inner.visit_str() { ret false; }
        self.bump_past::<str>();
        true
    }

    fn visit_estr_box() -> bool {
        self.align_to::<str/@>();
        if ! self.inner.visit_estr_box() { ret false; }
        self.bump_past::<str/@>();
        true
    }

    fn visit_estr_uniq() -> bool {
        self.align_to::<str/~>();
        if ! self.inner.visit_estr_uniq() { ret false; }
        self.bump_past::<str/~>();
        true
    }

    fn visit_estr_slice() -> bool {
        self.align_to::<str/&static>();
        if ! self.inner.visit_estr_slice() { ret false; }
        self.bump_past::<str/&static>();
        true
    }

    fn visit_estr_fixed(sz: uint) -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_estr_fixed(sz) { ret false; }
        self.bump(sz);
        true
    }

    fn visit_enter_box(mtbl: uint) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_enter_box(mtbl) { ret false; }
        true
    }

    fn visit_leave_box(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_box(mtbl) { ret false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_enter_uniq(mtbl: uint) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_enter_uniq(mtbl) { ret false; }
        true
    }

    fn visit_leave_uniq(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_uniq(mtbl) { ret false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_enter_ptr(mtbl: uint) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_enter_ptr(mtbl) { ret false; }
        true
    }

    fn visit_leave_ptr(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_ptr(mtbl) { ret false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_enter_rptr(mtbl: uint) -> bool {
        self.align_to::<&static.u8>();
        if ! self.inner.visit_enter_rptr(mtbl) { ret false; }
        true
    }

    fn visit_leave_rptr(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_rptr(mtbl) { ret false; }
        self.bump_past::<&static.u8>();
        true
    }

    fn visit_enter_vec(mtbl: uint) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_enter_vec(mtbl) { ret false; }
        true
    }

    fn visit_leave_vec(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_vec(mtbl) { ret false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_enter_evec_box(mtbl: uint) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_enter_evec_box(mtbl) { ret false; }
        true
    }

    fn visit_leave_evec_box(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_evec_box(mtbl) { ret false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_enter_evec_uniq(mtbl: uint) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_enter_evec_uniq(mtbl) { ret false; }
        true
    }

    fn visit_leave_evec_uniq(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_evec_uniq(mtbl) { ret false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_enter_evec_slice(mtbl: uint) -> bool {
        self.align_to::<&[u8]static>();
        if ! self.inner.visit_enter_evec_slice(mtbl) { ret false; }
        true
    }

    fn visit_leave_evec_slice(mtbl: uint) -> bool {
        if ! self.inner.visit_leave_evec_slice(mtbl) { ret false; }
        self.bump_past::<&[u8]static>();
        true
    }

    fn visit_enter_evec_fixed(mtbl: uint, n: uint,
                              sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_evec_fixed(mtbl, n, sz, align) {
            ret false;
        }
        true
    }

    fn visit_leave_evec_fixed(mtbl: uint, n: uint,
                              sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_evec_fixed(mtbl, n, sz, align) {
            ret false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_rec(n_fields, sz, align) { ret false; }
        true
    }

    fn visit_enter_rec_field(mtbl: uint, i: uint,
                             name: str/&) -> bool {
        if ! self.inner.visit_enter_rec_field(mtbl, i, name) { ret false; }
        true
    }

    fn visit_leave_rec_field(mtbl: uint, i: uint,
                             name: str/&) -> bool {
        if ! self.inner.visit_leave_rec_field(mtbl, i, name) { ret false; }
        true
    }

    fn visit_leave_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { ret false; }
        self.bump(sz);
        true
    }

    fn visit_enter_class(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(n_fields, sz, align) {
            ret false;
        }
        true
    }

    fn visit_enter_class_field(mtbl: uint, i: uint,
                               name: str/&) -> bool {
        if ! self.inner.visit_enter_class_field(mtbl, i, name) {
            ret false;
        }
        true
    }

    fn visit_leave_class_field(mtbl: uint, i: uint,
                               name: str/&) -> bool {
        if ! self.inner.visit_leave_class_field(mtbl, i, name) {
            ret false;
        }
        true
    }

    fn visit_leave_class(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_class(n_fields, sz, align) {
            ret false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { ret false; }
        true
    }

    fn visit_enter_tup_field(i: uint) -> bool {
        if ! self.inner.visit_enter_tup_field(i) { ret false; }
        true
    }

    fn visit_leave_tup_field(i: uint) -> bool {
        if ! self.inner.visit_leave_tup_field(i) { ret false; }
        true
    }

    fn visit_leave_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { ret false; }
        self.bump(sz);
        true
    }

    fn visit_enter_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            ret false;
        }
        true
    }

    fn visit_enter_fn_input(i: uint, mode: uint) -> bool {
        if ! self.inner.visit_enter_fn_input(i, mode) { ret false; }
        true
    }

    fn visit_leave_fn_input(i: uint, mode: uint) -> bool {
        if ! self.inner.visit_leave_fn_input(i, mode) { ret false; }
        true
    }

    fn visit_enter_fn_output(retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn_output(retstyle) { ret false; }
        true
    }

    fn visit_leave_fn_output(retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn_output(retstyle) { ret false; }
        true
    }

    fn visit_leave_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn(purity, proto, n_inputs, retstyle) {
            ret false;
        }
        true
    }

    fn visit_enter_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_enum(n_variants, sz, align) { ret false; }
        true
    }

    fn visit_enter_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: str/&) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            ret false;
        }
        true
    }

    fn visit_enter_enum_variant_field(i: uint) -> bool {
        if ! self.inner.visit_enter_enum_variant_field(i) { ret false; }
        true
    }

    fn visit_leave_enum_variant_field(i: uint) -> bool {
        if ! self.inner.visit_leave_enum_variant_field(i) { ret false; }
        true
    }

    fn visit_leave_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: str/&) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            ret false;
        }
        true
    }

    fn visit_leave_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_enum(n_variants, sz, align) { ret false; }
        self.bump(sz);
        true
    }

    fn visit_iface() -> bool {
        self.align_to::<ty_visitor>();
        if ! self.inner.visit_iface() { ret false; }
        self.bump_past::<ty_visitor>();
        true
    }

    fn visit_enter_res() -> bool {
        // FIXME: I _think_ a resource takes no space,
        // but I might be wrong.
        if ! self.inner.visit_enter_res() { ret false; }
        true
    }

    fn visit_leave_res() -> bool {
        if ! self.inner.visit_leave_res() { ret false; }
        true
    }

    fn visit_var() -> bool {
        if ! self.inner.visit_var() { ret false; }
        true
    }

    fn visit_var_integral() -> bool {
        if ! self.inner.visit_var_integral() { ret false; }
        true
    }

    fn visit_param(i: uint) -> bool {
        if ! self.inner.visit_param(i) { ret false; }
        true
    }

    fn visit_self() -> bool {
        self.align_to::<&static.u8>();
        if ! self.inner.visit_self() { ret false; }
        self.align_to::<&static.u8>();
        true
    }

    fn visit_type() -> bool {
        if ! self.inner.visit_type() { ret false; }
        true
    }

    fn visit_opaque_box() -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_opaque_box() { ret false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_enter_constr() -> bool {
        if ! self.inner.visit_enter_constr() { ret false; }
        true
    }

    fn visit_leave_constr() -> bool {
        if ! self.inner.visit_leave_constr() { ret false; }
        true
    }

    fn visit_closure_ptr(ck: uint) -> bool {
        self.align_to::<fn@()>();
        if ! self.inner.visit_closure_ptr(ck) { ret false; }
        self.bump_past::<fn@()>();
        true
    }
}

enum my_visitor = @{
    mut ptr1: *c_void,
    mut ptr2: *c_void,
    mut vals: ~[str]
};

impl extra_methods for my_visitor {
    fn get<T>(f: fn(T)) {
        unsafe {
            f(*(self.ptr1 as *T));
        }
    }
}

impl of movable_ptr for my_visitor {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void) {
        self.ptr1 = adjustment(self.ptr1);
        self.ptr2 = adjustment(self.ptr2);
    }
}

impl of ty_visitor for my_visitor {

    fn visit_bot() -> bool { true }
    fn visit_nil() -> bool { true }
    fn visit_bool() -> bool {
/*
        self.get::<bool>() {|b|
            self.vals += ~[bool::to_str(b)];
        }
*/
        true
    }
    fn visit_int() -> bool {
/*
        self.get::<int>() {|i|
            self.vals += ~[int::to_str(i, 10u)];
        }
*/
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
    fn visit_enter_evec_fixed(_mtbl: uint, _n: uint,
                              _sz: uint, _align: uint) -> bool { true }
    fn visit_leave_evec_fixed(_mtbl: uint, _n: uint,
                              _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_rec_field(_mtbl: uint, _i: uint,
                             _name: str/&) -> bool { true }
    fn visit_leave_rec_field(_mtbl: uint, _i: uint,
                             _name: str/&) -> bool { true }
    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_class_field(_mtbl: uint, _i: uint,
                               _name: str/&) -> bool { true }
    fn visit_leave_class_field(_mtbl: uint, _i: uint,
                               _name: str/&) -> bool { true }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_enter_tup_field(_i: uint) -> bool { true }
    fn visit_leave_tup_field(_i: uint) -> bool { true }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_enter_fn_input(_i: uint, _mode: uint) -> bool { true }
    fn visit_leave_fn_input(_i: uint, _mode: uint) -> bool { true }
    fn visit_enter_fn_output(_retstyle: uint) -> bool { true }
    fn visit_leave_fn_output(_retstyle: uint) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }
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
    fn visit_leave_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }

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
    let u = my_visitor(@{mut ptr1: p,
                         mut ptr2: p,
                         mut vals: ~[]});
    let v = ptr_visit_adaptor({inner: u});
    let vv = v as intrinsic::ty_visitor;
    intrinsic::visit_ty::<(int,int,int,bool,bool)>(vv);

    for (copy u.vals).each {|s|
        io::println(#fmt("val: %s", s));
    }
    assert u.vals == ["1", "2", "3", "true", "false"];
 }
