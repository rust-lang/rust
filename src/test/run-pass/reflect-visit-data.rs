
import intrinsic::{tydesc, get_tydesc, visit_tydesc, ty_visitor};
import libc::c_void;

// FIXME: this is a near-duplicate of code in core::vec.
type unboxed_vec_repr = {
    mut fill: uint,
    mut alloc: uint,
    data: u8
};

#[doc = "High-level interfaces to `intrinsic::visit_ty` reflection system."]

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
      do self.inner.move_ptr() |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    fn align(a: uint) {
      do self.inner.move_ptr() |p| {
            align(p as uint, a) as *c_void
      };
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
        self.align_to::<~str>();
        if ! self.inner.visit_str() { ret false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_box() -> bool {
        self.align_to::<@str>();
        if ! self.inner.visit_estr_box() { ret false; }
        self.bump_past::<@str>();
        true
    }

    fn visit_estr_uniq() -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_estr_uniq() { ret false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_slice() -> bool {
        self.align_to::<&static/str>();
        if ! self.inner.visit_estr_slice() { ret false; }
        self.bump_past::<&static/str>();
        true
    }

    fn visit_estr_fixed(n: uint,
                        sz: uint,
                        align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_estr_fixed(n, sz, align) { ret false; }
        self.bump(sz);
        true
    }

    fn visit_box(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_box(mtbl, inner) { ret false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_uniq(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq(mtbl, inner) { ret false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_ptr(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_ptr(mtbl, inner) { ret false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_rptr(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<&static/u8>();
        if ! self.inner.visit_rptr(mtbl, inner) { ret false; }
        self.bump_past::<&static/u8>();
        true
    }

    fn visit_unboxed_vec(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<unboxed_vec_repr>();
        // FIXME: Inner really has to move its own pointers on this one.
        // or else possibly we could have some weird interface wherein we
        // read-off a word from inner's pointers, but the read-word has to
        // always be the same in all sub-pointers? Dubious.
        if ! self.inner.visit_vec(mtbl, inner) { ret false; }
        true
    }

    fn visit_vec(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_vec(mtbl, inner) { ret false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_box(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_evec_box(mtbl, inner) { ret false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_evec_uniq(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_evec_uniq(mtbl, inner) { ret false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_slice(mtbl: uint, inner: *tydesc) -> bool {
        self.align_to::<&static/[u8]>();
        if ! self.inner.visit_evec_slice(mtbl, inner) { ret false; }
        self.bump_past::<&static/[u8]>();
        true
    }

    fn visit_evec_fixed(n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *tydesc) -> bool {
        self.align(align);
        if ! self.inner.visit_evec_fixed(n, sz, align, mtbl, inner) {
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

    fn visit_rec_field(i: uint, name: &str,
                       mtbl: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_rec_field(i, name, mtbl, inner) { ret false; }
        true
    }

    fn visit_leave_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { ret false; }
        true
    }

    fn visit_enter_class(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(n_fields, sz, align) {
            ret false;
        }
        true
    }

    fn visit_class_field(i: uint, name: &str,
                         mtbl: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_class_field(i, name, mtbl, inner) {
            ret false;
        }
        true
    }

    fn visit_leave_class(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_class(n_fields, sz, align) {
            ret false;
        }
        true
    }

    fn visit_enter_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { ret false; }
        true
    }

    fn visit_tup_field(i: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_tup_field(i, inner) { ret false; }
        true
    }

    fn visit_leave_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { ret false; }
        true
    }

    fn visit_enter_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            ret false
        }
        true
    }

    fn visit_fn_input(i: uint, mode: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_fn_input(i, mode, inner) { ret false; }
        true
    }

    fn visit_fn_output(retstyle: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_fn_output(retstyle, inner) { ret false; }
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
                                name: &str) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            ret false;
        }
        true
    }

    fn visit_enum_variant_field(i: uint, inner: *tydesc) -> bool {
        if ! self.inner.visit_enum_variant_field(i, inner) { ret false; }
        true
    }

    fn visit_leave_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            ret false;
        }
        true
    }

    fn visit_leave_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_enum(n_variants, sz, align) { ret false; }
        true
    }

    fn visit_trait() -> bool {
        self.align_to::<ty_visitor>();
        if ! self.inner.visit_trait() { ret false; }
        self.bump_past::<ty_visitor>();
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
        self.align_to::<&static/u8>();
        if ! self.inner.visit_self() { ret false; }
        self.align_to::<&static/u8>();
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

    fn visit_constr(inner: *tydesc) -> bool {
        if ! self.inner.visit_constr(inner) { ret false; }
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
    mut vals: ~[~str]
};

impl extra_methods for my_visitor {
    fn get<T>(f: fn(T)) {
        unsafe {
            f(*(self.ptr1 as *T));
        }
    }

    fn visit_inner(inner: *tydesc) -> bool {
        let u = my_visitor(*self);
        let v = ptr_visit_adaptor({inner: u});
        visit_tydesc(inner, v as ty_visitor);
        true
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
      do self.get::<bool>() |b| {
            self.vals += ~[bool::to_str(b)];
      };
      true
    }
    fn visit_int() -> bool {
      do self.get::<int>() |i| {
            self.vals += ~[int::to_str(i, 10u)];
      };
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
    fn visit_estr_fixed(_n: uint, _sz: uint,
                        _align: uint) -> bool { true }

    fn visit_box(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_uniq(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_ptr(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_rptr(_mtbl: uint, _inner: *tydesc) -> bool { true }

    fn visit_vec(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_unboxed_vec(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_evec_box(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_evec_uniq(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_evec_slice(_mtbl: uint, _inner: *tydesc) -> bool { true }
    fn visit_evec_fixed(_n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *tydesc) -> bool { true }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_rec_field(_i: uint, _name: &str,
                       _mtbl: uint, inner: *tydesc) -> bool {
        #error("rec field!");
        self.visit_inner(inner)
    }
    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(_i: uint, _name: &str,
                         _mtbl: uint, inner: *tydesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }
    fn visit_tup_field(_i: uint, inner: *tydesc) -> bool {
        #error("tup field!");
        self.visit_inner(inner)
    }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        // FIXME: this needs to rewind between enum variants, or something.
        true
    }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(_i: uint, inner: *tydesc) -> bool {
        self.visit_inner(inner)
    }
    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_leave_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(_i: uint, _mode: uint, _inner: *tydesc) -> bool { true }
    fn visit_fn_output(_retstyle: uint, _inner: *tydesc) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait() -> bool { true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }
    fn visit_opaque_box() -> bool { true }
    fn visit_constr(_inner: *tydesc) -> bool { true }
    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

fn get_tydesc_for<T>(&&_t: T) -> *tydesc {
    get_tydesc::<T>()
}

fn main() {
    let r = (1,2,3,true,false,{x:5,y:4,z:3});
    let p = ptr::addr_of(r) as *c_void;
    let u = my_visitor(@{mut ptr1: p,
                         mut ptr2: p,
                         mut vals: ~[]});
    let v = ptr_visit_adaptor({inner: u});
    let td = get_tydesc_for(r);
    unsafe { #error("tydesc sz: %u, align: %u",
                    (*td).size, (*td).align); }
    let v = v as ty_visitor;
    visit_tydesc(td, v);

    for (copy u.vals).each |s| {
        io::println(#fmt("val: %s", s));
    }
    #error("%?", copy u.vals);
    assert u.vals == ~[~"1", ~"2", ~"3", ~"true", ~"false", ~"5", ~"4", ~"3"];
 }
