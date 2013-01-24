// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

More runtime type reflection

*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use cast::transmute;
use cast;
use char;
use dvec::DVec;
use intrinsic;
use intrinsic::{TyDesc, TyVisitor, visit_tydesc};
use io;
use io::{Writer, WriterUtil};
use libc::c_void;
use managed;
use managed::raw::BoxHeaderRepr;
use ptr;
use reflect;
use reflect::{MovePtr, MovePtrAdaptor, align};
use repr;
use str;
use sys;
use sys::TypeDesc;
use to_str::ToStr;
use uint;
use vec::UnboxedVecRepr;
use vec::raw::{VecRepr, SliceRepr};
use vec;

pub use managed::raw::BoxRepr;

/// Helpers

trait EscapedCharWriter {
    fn write_escaped_char(ch: char);
}

impl Writer : EscapedCharWriter {
    fn write_escaped_char(ch: char) {
        match ch {
            '\t' => self.write_str("\\t"),
            '\r' => self.write_str("\\r"),
            '\n' => self.write_str("\\n"),
            '\\' => self.write_str("\\\\"),
            '\'' => self.write_str("\\'"),
            '"' => self.write_str("\\\""),
            '\x20'..'\x7e' => self.write_char(ch),
            _ => {
                // FIXME #4423: This is inefficient because it requires a
                // malloc.
                self.write_str(char::escape_unicode(ch))
            }
        }
    }
}

/// Representations

trait Repr {
    fn write_repr(writer: @Writer);
}

impl () : Repr {
    fn write_repr(writer: @Writer) { writer.write_str("()"); }
}

impl bool : Repr {
    fn write_repr(writer: @Writer) {
        writer.write_str(if self { "true" } else { "false" })
    }
}

impl int : Repr {
    fn write_repr(writer: @Writer) { writer.write_int(self); }
}
impl i8 : Repr {
    fn write_repr(writer: @Writer) { writer.write_int(self as int); }
}
impl i16 : Repr {
    fn write_repr(writer: @Writer) { writer.write_int(self as int); }
}
impl i32 : Repr {
    fn write_repr(writer: @Writer) { writer.write_int(self as int); }
}
impl i64 : Repr {
    // FIXME #4424: This can lose precision.
    fn write_repr(writer: @Writer) { writer.write_int(self as int); }
}

impl uint : Repr {
    fn write_repr(writer: @Writer) { writer.write_uint(self); }
}
impl u8 : Repr {
    fn write_repr(writer: @Writer) { writer.write_uint(self as uint); }
}
impl u16 : Repr {
    fn write_repr(writer: @Writer) { writer.write_uint(self as uint); }
}
impl u32 : Repr {
    fn write_repr(writer: @Writer) { writer.write_uint(self as uint); }
}
impl u64 : Repr {
    // FIXME #4424: This can lose precision.
    fn write_repr(writer: @Writer) { writer.write_uint(self as uint); }
}

impl float : Repr {
    // FIXME #4423: This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}
impl f32 : Repr {
    // FIXME #4423 This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}
impl f64 : Repr {
    // FIXME #4423: This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}

impl char : Repr {
    fn write_repr(writer: @Writer) { writer.write_char(self); }
}


// New implementation using reflect::MovePtr

enum VariantState {
    Degenerate,
    TagMatch,
    TagMismatch,
}

pub struct ReprVisitor {
    mut ptr: *c_void,
    ptr_stk: DVec<*c_void>,
    var_stk: DVec<VariantState>,
    writer: @Writer
}
pub fn ReprVisitor(ptr: *c_void, writer: @Writer) -> ReprVisitor {
    ReprVisitor { ptr: ptr,
                  ptr_stk: DVec(),
                  var_stk: DVec(),
                  writer: writer }
}

impl ReprVisitor : MovePtr {
    #[inline(always)]
    fn move_ptr(adjustment: fn(*c_void) -> *c_void) {
        self.ptr = adjustment(self.ptr);
    }
    fn push_ptr() {
        self.ptr_stk.push(self.ptr);
    }
    fn pop_ptr() {
        self.ptr = self.ptr_stk.pop();
    }
}

impl ReprVisitor {

    // Various helpers for the TyVisitor impl

    #[inline(always)]
    fn get<T>(f: fn(&T)) -> bool {
        unsafe {
            f(transmute::<*c_void,&T>(copy self.ptr));
        }
        true
    }

    #[inline(always)]
    fn bump(sz: uint) {
      do self.move_ptr() |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    fn bump_past<T>() {
        self.bump(sys::size_of::<T>());
    }

    #[inline(always)]
    fn visit_inner(inner: *TyDesc) -> bool {
        self.visit_ptr_inner(self.ptr, inner)
    }

    #[inline(always)]
    fn visit_ptr_inner(ptr: *c_void, inner: *TyDesc) -> bool {
        unsafe {
            let mut u = ReprVisitor(ptr, self.writer);
            let v = reflect::MovePtrAdaptor(move u);
            visit_tydesc(inner, (move v) as @TyVisitor);
            true
        }
    }

    #[inline(always)]
    fn write<T:Repr>() -> bool {
        do self.get |v:&T| {
            v.write_repr(self.writer);
        }
    }

    fn write_escaped_slice(slice: &str) {
        self.writer.write_char('"');
        for str::chars_each(slice) |ch| {
            self.writer.write_escaped_char(ch);
        }
        self.writer.write_char('"');
    }

    fn write_mut_qualifier(mtbl: uint) {
        if mtbl == 0 {
            self.writer.write_str("mut ");
        } else if mtbl == 1 {
            // skip, this is ast::m_imm
        } else {
            assert mtbl == 2;
            self.writer.write_str("const ");
        }
    }

    fn write_vec_range(mtbl: uint, ptr: *u8, len: uint,
                       inner: *TyDesc) -> bool {
        let mut p = ptr;
        let end = ptr::offset(p, len);
        let (sz, al) = unsafe { ((*inner).size, (*inner).align) };
        self.writer.write_char('[');
        let mut first = true;
        while p as uint < end as uint {
            if first {
                first = false;
            } else {
                self.writer.write_str(", ");
            }
            self.write_mut_qualifier(mtbl);
            self.visit_ptr_inner(p as *c_void, inner);
            p = align(ptr::offset(p, sz) as uint, al) as *u8;
        }
        self.writer.write_char(']');
        true
    }

    fn write_unboxed_vec_repr(mtbl: uint, v: &UnboxedVecRepr,
                              inner: *TyDesc) -> bool {
        self.write_vec_range(mtbl, ptr::to_unsafe_ptr(&v.data),
                             v.fill, inner)
    }


}

impl ReprVisitor : TyVisitor {
    fn visit_bot() -> bool {
        self.writer.write_str("!");
        true
    }
    fn visit_nil() -> bool { self.write::<()>() }
    fn visit_bool() -> bool { self.write::<bool>() }
    fn visit_int() -> bool { self.write::<int>() }
    fn visit_i8() -> bool { self.write::<i8>() }
    fn visit_i16() -> bool { self.write::<i16>() }
    fn visit_i32() -> bool { self.write::<i32>()  }
    fn visit_i64() -> bool { self.write::<i64>() }

    fn visit_uint() -> bool { self.write::<uint>() }
    fn visit_u8() -> bool { self.write::<u8>() }
    fn visit_u16() -> bool { self.write::<u16>() }
    fn visit_u32() -> bool { self.write::<u32>() }
    fn visit_u64() -> bool { self.write::<u64>() }

    fn visit_float() -> bool { self.write::<float>() }
    fn visit_f32() -> bool { self.write::<f32>() }
    fn visit_f64() -> bool { self.write::<f64>() }

    fn visit_char() -> bool {
        do self.get::<char> |&ch| {
            self.writer.write_char('\'');
            self.writer.write_escaped_char(ch);
            self.writer.write_char('\'');
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_str() -> bool { fail; }

    fn visit_estr_box() -> bool {
        do self.get::<@str> |s| {
            self.writer.write_char('@');
            self.write_escaped_slice(*s);
        }
    }
    fn visit_estr_uniq() -> bool {
        do self.get::<~str> |s| {
            self.writer.write_char('~');
            self.write_escaped_slice(*s);
        }
    }
    fn visit_estr_slice() -> bool {
        do self.get::<&str> |s| {
            self.write_escaped_slice(*s);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_estr_fixed(_n: uint, _sz: uint,
                        _align: uint) -> bool { fail; }

    fn visit_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('@');
        self.write_mut_qualifier(mtbl);
        do self.get::<&managed::raw::BoxRepr> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, inner);
        }
    }

    fn visit_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('~');
        self.write_mut_qualifier(mtbl);
        do self.get::<&managed::raw::BoxRepr> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, inner);
        }
    }

    fn visit_ptr(_mtbl: uint, _inner: *TyDesc) -> bool {
        do self.get::<*c_void> |p| {
            self.writer.write_str(fmt!("(0x%x as *())",
                                       *p as uint));
        }
    }

    fn visit_rptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('&');
        self.write_mut_qualifier(mtbl);
        do self.get::<*c_void> |p| {
            self.visit_ptr_inner(*p, inner);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_vec(_mtbl: uint, _inner: *TyDesc) -> bool { fail; }


    fn visit_unboxed_vec(mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<vec::UnboxedVecRepr> |b| {
            self.write_unboxed_vec_repr(mtbl, b, inner);
        }
    }

    fn visit_evec_box(mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<&VecRepr> |b| {
            self.writer.write_char('@');
            self.write_unboxed_vec_repr(mtbl, &b.unboxed, inner);
        }
    }

    fn visit_evec_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<&VecRepr> |b| {
            self.writer.write_char('~');
            self.write_unboxed_vec_repr(mtbl, &b.unboxed, inner);
        }
    }

    fn visit_evec_slice(mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<SliceRepr> |s| {
            self.writer.write_char('&');
            self.write_vec_range(mtbl, s.data, s.len, inner);
        }
    }

    fn visit_evec_fixed(_n: uint, sz: uint, _align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<u8> |b| {
            self.write_vec_range(mtbl, ptr::to_unsafe_ptr(b), sz, inner);
        }
    }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('{');
        true
    }

    fn visit_rec_field(i: uint, name: &str,
                       mtbl: uint, inner: *TyDesc) -> bool {
        if i != 0 {
            self.writer.write_str(", ");
        }
        self.write_mut_qualifier(mtbl);
        self.writer.write_str(name);
        self.writer.write_str(": ");
        self.visit_inner(inner);
        true
    }

    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('}');
        true
    }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        self.writer.write_char('{');
        true
    }
    fn visit_class_field(i: uint, name: &str,
                         mtbl: uint, inner: *TyDesc) -> bool {
        if i != 0 {
            self.writer.write_str(", ");
        }
        self.write_mut_qualifier(mtbl);
        self.writer.write_str(name);
        self.writer.write_str(": ");
        self.visit_inner(inner);
        true
    }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        self.writer.write_char('}');
        true
    }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('(');
        true
    }
    fn visit_tup_field(i: uint, inner: *TyDesc) -> bool {
        if i != 0 {
            self.writer.write_str(", ");
        }
        self.visit_inner(inner);
        true
    }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char(')');
        true
    }

    fn visit_enter_enum(n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        if n_variants == 1 {
            self.var_stk.push(Degenerate)
        } else {
            self.var_stk.push(TagMatch)
        }
        true
    }
    fn visit_enter_enum_variant(_variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        let mut write = false;
        match self.var_stk.pop() {
            Degenerate => {
                write = true;
                self.var_stk.push(Degenerate);
            }
            TagMatch | TagMismatch => {
                do self.get::<int>() |t| {
                    if disr_val == *t {
                        write = true;
                        self.var_stk.push(TagMatch);
                    } else {
                        self.var_stk.push(TagMismatch);
                    }
                };
                self.bump_past::<int>();
            }
        }

        if write {
            self.writer.write_str(name);
            if n_fields > 0 {
                self.writer.write_char('(');
            }
        }
        true
    }
    fn visit_enum_variant_field(i: uint, inner: *TyDesc) -> bool {
        match self.var_stk.last() {
            Degenerate | TagMatch => {
                if i != 0 {
                    self.writer.write_str(", ");
                }
                if ! self.visit_inner(inner) {
                    return false;
                }
            }
            TagMismatch => ()
        }
        true
    }
    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                n_fields: uint,
                                _name: &str) -> bool {
        match self.var_stk.last() {
            Degenerate | TagMatch => {
                if n_fields > 0 {
                    self.writer.write_char(')');
                }
            }
            TagMismatch => ()
        }
        true
    }
    fn visit_leave_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        self.var_stk.pop();
        true
    }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(_i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(_retstyle: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait() -> bool { true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }

    fn visit_opaque_box() -> bool {
        self.writer.write_char('@');
        do self.get::<&managed::raw::BoxRepr> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, b.header.type_desc);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_constr(_inner: *TyDesc) -> bool { fail; }

    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

pub fn write_repr<T>(writer: @Writer, object: &T) {
    unsafe {
        let ptr = ptr::to_unsafe_ptr(object) as *c_void;
        let tydesc = intrinsic::get_tydesc::<T>();
        let mut u = ReprVisitor(ptr, writer);
        let v = reflect::MovePtrAdaptor(move u);
        visit_tydesc(tydesc, (move v) as @TyVisitor)
    }
}

#[test]
fn test_repr() {

    fn exact_test<T>(t: &T, e:&str) {
        let s : &str = io::with_str_writer(|w| repr::write_repr(w, t));
        if s != e {
            error!("expected '%s', got '%s'",
                   e, s);
        }
        assert s == e;
    }


    exact_test(&10, "10");
    exact_test(&true, "true");
    exact_test(&false, "false");
    exact_test(&1.234, "1.2340");
    exact_test(&(&"hello"), "\"hello\"");
    exact_test(&(@"hello"), "@\"hello\"");
    exact_test(&(~"he\u10f3llo"), "~\"he\\u10f3llo\"");

    // FIXME #4210: the mut fields are a bit off here.
    exact_test(&(@10), "@10");
    exact_test(&(@mut 10), "@10");
    exact_test(&(~10), "~10");
    exact_test(&(~mut 10), "~mut 10");
    exact_test(&(&10), "&10");
    let mut x = 10;
    exact_test(&(&mut x), "&mut 10");

    exact_test(&(@[1,2,3,4,5,6,7,8]),
               "@[1, 2, 3, 4, 5, 6, 7, 8]");
    exact_test(&(@[1u8,2u8,3u8,4u8]),
               "@[1, 2, 3, 4]");
    exact_test(&(@["hi", "there"]),
               "@[\"hi\", \"there\"]");
    exact_test(&(~["hi", "there"]),
               "~[\"hi\", \"there\"]");
    exact_test(&(&["hi", "there"]),
               "&[\"hi\", \"there\"]");
    exact_test(&({a:10, b:1.234}),
               "{a: 10, b: 1.2340}");
    exact_test(&(@{a:10, b:1.234}),
               "@{a: 10, b: 1.2340}");
    exact_test(&(~{a:10, b:1.234}),
               "~{a: 10, b: 1.2340}");
    exact_test(&(10_u8, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u16, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u32, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u64, ~"hello"),
               "(10, ~\"hello\")");
}
