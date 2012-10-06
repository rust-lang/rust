/*!

Runtime type reflection

*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use intrinsic::{TyDesc, get_tydesc, visit_tydesc, TyVisitor};
use libc::c_void;

/**
 * Trait for visitor that wishes to reflect on data. To use this, create a
 * struct that encapsulates the set of pointers you wish to walk through a
 * data structure, and implement both `MovePtr` for it as well as `TyVisitor`;
 * then build a MovePtrAdaptor wrapped around your struct.
 */
pub trait MovePtr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void);
}

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

/// Adaptor to wrap around visitors implementing MovePtr.
struct MovePtrAdaptor<V: TyVisitor MovePtr> {
    inner: V
}
pub fn MovePtrAdaptor<V: TyVisitor MovePtr>(v: V) -> MovePtrAdaptor<V> {
    MovePtrAdaptor { inner: move v }
}

/// Abstract type-directed pointer-movement using the MovePtr trait
impl<V: TyVisitor MovePtr> MovePtrAdaptor<V>: TyVisitor {

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
        if ! self.inner.visit_bot() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_nil() -> bool {
        self.align_to::<()>();
        if ! self.inner.visit_nil() { return false; }
        self.bump_past::<()>();
        true
    }

    fn visit_bool() -> bool {
        self.align_to::<bool>();
        if ! self.inner.visit_bool() { return false; }
        self.bump_past::<bool>();
        true
    }

    fn visit_int() -> bool {
        self.align_to::<int>();
        if ! self.inner.visit_int() { return false; }
        self.bump_past::<int>();
        true
    }

    fn visit_i8() -> bool {
        self.align_to::<i8>();
        if ! self.inner.visit_i8() { return false; }
        self.bump_past::<i8>();
        true
    }

    fn visit_i16() -> bool {
        self.align_to::<i16>();
        if ! self.inner.visit_i16() { return false; }
        self.bump_past::<i16>();
        true
    }

    fn visit_i32() -> bool {
        self.align_to::<i32>();
        if ! self.inner.visit_i32() { return false; }
        self.bump_past::<i32>();
        true
    }

    fn visit_i64() -> bool {
        self.align_to::<i64>();
        if ! self.inner.visit_i64() { return false; }
        self.bump_past::<i64>();
        true
    }

    fn visit_uint() -> bool {
        self.align_to::<uint>();
        if ! self.inner.visit_uint() { return false; }
        self.bump_past::<uint>();
        true
    }

    fn visit_u8() -> bool {
        self.align_to::<u8>();
        if ! self.inner.visit_u8() { return false; }
        self.bump_past::<u8>();
        true
    }

    fn visit_u16() -> bool {
        self.align_to::<u16>();
        if ! self.inner.visit_u16() { return false; }
        self.bump_past::<u16>();
        true
    }

    fn visit_u32() -> bool {
        self.align_to::<u32>();
        if ! self.inner.visit_u32() { return false; }
        self.bump_past::<u32>();
        true
    }

    fn visit_u64() -> bool {
        self.align_to::<u64>();
        if ! self.inner.visit_u64() { return false; }
        self.bump_past::<u64>();
        true
    }

    fn visit_float() -> bool {
        self.align_to::<float>();
        if ! self.inner.visit_float() { return false; }
        self.bump_past::<float>();
        true
    }

    fn visit_f32() -> bool {
        self.align_to::<f32>();
        if ! self.inner.visit_f32() { return false; }
        self.bump_past::<f32>();
        true
    }

    fn visit_f64() -> bool {
        self.align_to::<f64>();
        if ! self.inner.visit_f64() { return false; }
        self.bump_past::<f64>();
        true
    }

    fn visit_char() -> bool {
        self.align_to::<char>();
        if ! self.inner.visit_char() { return false; }
        self.bump_past::<char>();
        true
    }

    fn visit_str() -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_str() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_box() -> bool {
        self.align_to::<@str>();
        if ! self.inner.visit_estr_box() { return false; }
        self.bump_past::<@str>();
        true
    }

    fn visit_estr_uniq() -> bool {
        self.align_to::<~str>();
        if ! self.inner.visit_estr_uniq() { return false; }
        self.bump_past::<~str>();
        true
    }

    fn visit_estr_slice() -> bool {
        self.align_to::<&static/str>();
        if ! self.inner.visit_estr_slice() { return false; }
        self.bump_past::<&static/str>();
        true
    }

    fn visit_estr_fixed(n: uint,
                        sz: uint,
                        align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_estr_fixed(n, sz, align) { return false; }
        self.bump(sz);
        true
    }

    fn visit_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_box(mtbl, inner) { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~u8>();
        if ! self.inner.visit_uniq(mtbl, inner) { return false; }
        self.bump_past::<~u8>();
        true
    }

    fn visit_ptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<*u8>();
        if ! self.inner.visit_ptr(mtbl, inner) { return false; }
        self.bump_past::<*u8>();
        true
    }

    fn visit_rptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&static/u8>();
        if ! self.inner.visit_rptr(mtbl, inner) { return false; }
        self.bump_past::<&static/u8>();
        true
    }

    fn visit_unboxed_vec(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<vec::raw::UnboxedVecRepr>();
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        true
    }

    fn visit_vec(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_vec(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<@[u8]>();
        if ! self.inner.visit_evec_box(mtbl, inner) { return false; }
        self.bump_past::<@[u8]>();
        true
    }

    fn visit_evec_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<~[u8]>();
        if ! self.inner.visit_evec_uniq(mtbl, inner) { return false; }
        self.bump_past::<~[u8]>();
        true
    }

    fn visit_evec_slice(mtbl: uint, inner: *TyDesc) -> bool {
        self.align_to::<&static/[u8]>();
        if ! self.inner.visit_evec_slice(mtbl, inner) { return false; }
        self.bump_past::<&static/[u8]>();
        true
    }

    fn visit_evec_fixed(n: uint, sz: uint, align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        self.align(align);
        if ! self.inner.visit_evec_fixed(n, sz, align, mtbl, inner) {
            return false;
        }
        self.bump(sz);
        true
    }

    fn visit_enter_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_rec_field(i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool {
        unsafe { self.align((*inner).align); }
        if ! self.inner.visit_rec_field(i, name, mtbl, inner) {
            return false;
        }
        unsafe { self.bump((*inner).size); }
        true
    }

    fn visit_leave_rec(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_rec(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_class(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_class_field(i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool {
        unsafe { self.align((*inner).align); }
        if ! self.inner.visit_class_field(i, name, mtbl, inner) {
            return false;
        }
        unsafe { self.bump((*inner).size); }
        true
    }

    fn visit_leave_class(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_class(n_fields, sz, align) {
            return false;
        }
        true
    }

    fn visit_enter_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_tup_field(i: uint, inner: *TyDesc) -> bool {
        unsafe { self.align((*inner).align); }
        if ! self.inner.visit_tup_field(i, inner) { return false; }
        unsafe { self.bump((*inner).size); }
        true
    }

    fn visit_leave_tup(n_fields: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_tup(n_fields, sz, align) { return false; }
        true
    }

    fn visit_enter_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_enter_fn(purity, proto, n_inputs, retstyle) {
            return false
        }
        true
    }

    fn visit_fn_input(i: uint, mode: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_input(i, mode, inner) { return false; }
        true
    }

    fn visit_fn_output(retstyle: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_fn_output(retstyle, inner) { return false; }
        true
    }

    fn visit_leave_fn(purity: uint, proto: uint,
                      n_inputs: uint, retstyle: uint) -> bool {
        if ! self.inner.visit_leave_fn(purity, proto, n_inputs, retstyle) {
            return false;
        }
        true
    }

    fn visit_enter_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        self.align(align);
        if ! self.inner.visit_enter_enum(n_variants, sz, align) {
            return false;
        }
        true
    }

    fn visit_enter_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_enter_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_enum_variant_field(i: uint, inner: *TyDesc) -> bool {
        if ! self.inner.visit_enum_variant_field(i, inner) { return false; }
        true
    }

    fn visit_leave_enum_variant(variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        if ! self.inner.visit_leave_enum_variant(variant, disr_val,
                                                 n_fields, name) {
            return false;
        }
        true
    }

    fn visit_leave_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        if ! self.inner.visit_leave_enum(n_variants, sz, align) {
            return false;
        }
        true
    }

    fn visit_trait() -> bool {
        self.align_to::<TyVisitor>();
        if ! self.inner.visit_trait() { return false; }
        self.bump_past::<TyVisitor>();
        true
    }

    fn visit_var() -> bool {
        if ! self.inner.visit_var() { return false; }
        true
    }

    fn visit_var_integral() -> bool {
        if ! self.inner.visit_var_integral() { return false; }
        true
    }

    fn visit_param(i: uint) -> bool {
        if ! self.inner.visit_param(i) { return false; }
        true
    }

    fn visit_self() -> bool {
        self.align_to::<&static/u8>();
        if ! self.inner.visit_self() { return false; }
        self.align_to::<&static/u8>();
        true
    }

    fn visit_type() -> bool {
        if ! self.inner.visit_type() { return false; }
        true
    }

    fn visit_opaque_box() -> bool {
        self.align_to::<@u8>();
        if ! self.inner.visit_opaque_box() { return false; }
        self.bump_past::<@u8>();
        true
    }

    fn visit_constr(inner: *TyDesc) -> bool {
        if ! self.inner.visit_constr(inner) { return false; }
        true
    }

    fn visit_closure_ptr(ck: uint) -> bool {
        self.align_to::<fn@()>();
        if ! self.inner.visit_closure_ptr(ck) { return false; }
        self.bump_past::<fn@()>();
        true
    }
}
