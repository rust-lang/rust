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

#[allow(missing_doc)];

use cast::transmute;
use char;
use container::Container;
use io::{Writer, WriterUtil};
use iterator::Iterator;
use libc::c_void;
use option::{Some, None};
use ptr;
use reflect;
use reflect::{MovePtr, align};
use str::StrSlice;
use to_str::ToStr;
use vec::OwnedVector;
use unstable::intrinsics::{Opaque, TyDesc, TyVisitor, get_tydesc, visit_tydesc};
use unstable::raw;

#[cfg(test)] use io;

/// Helpers

trait EscapedCharWriter {
    fn write_escaped_char(&self, ch: char);
}

impl EscapedCharWriter for @Writer {
    fn write_escaped_char(&self, ch: char) {
        match ch {
            '\t' => self.write_str("\\t"),
            '\r' => self.write_str("\\r"),
            '\n' => self.write_str("\\n"),
            '\\' => self.write_str("\\\\"),
            '\'' => self.write_str("\\'"),
            '"' => self.write_str("\\\""),
            '\x20'..'\x7e' => self.write_char(ch),
            _ => {
                do char::escape_unicode(ch) |c| {
                    self.write_char(c);
                }
            }
        }
    }
}

/// Representations

trait Repr {
    fn write_repr(&self, writer: @Writer);
}

impl Repr for () {
    fn write_repr(&self, writer: @Writer) { writer.write_str("()"); }
}

impl Repr for bool {
    fn write_repr(&self, writer: @Writer) {
        writer.write_str(if *self { "true" } else { "false" })
    }
}

macro_rules! int_repr(($ty:ident) => (impl Repr for $ty {
    fn write_repr(&self, writer: @Writer) {
        do ::$ty::to_str_bytes(*self, 10u) |bits| {
            writer.write(bits);
        }
    }
}))

int_repr!(int)
int_repr!(i8)
int_repr!(i16)
int_repr!(i32)
int_repr!(i64)
int_repr!(uint)
int_repr!(u8)
int_repr!(u16)
int_repr!(u32)
int_repr!(u64)

macro_rules! num_repr(($ty:ident) => (impl Repr for $ty {
    fn write_repr(&self, writer: @Writer) {
        let s = self.to_str();
        writer.write(s.as_bytes());
    }
}))

num_repr!(float)
num_repr!(f32)
num_repr!(f64)

// New implementation using reflect::MovePtr

enum VariantState {
    SearchingFor(int),
    Matched,
    AlreadyFound
}

pub struct ReprVisitor {
    ptr: @mut *c_void,
    ptr_stk: @mut ~[*c_void],
    var_stk: @mut ~[VariantState],
    writer: @Writer
}
pub fn ReprVisitor(ptr: *c_void, writer: @Writer) -> ReprVisitor {
    ReprVisitor {
        ptr: @mut ptr,
        ptr_stk: @mut ~[],
        var_stk: @mut ~[],
        writer: writer,
    }
}

impl MovePtr for ReprVisitor {
    #[inline]
    fn move_ptr(&self, adjustment: &fn(*c_void) -> *c_void) {
        *self.ptr = adjustment(*self.ptr);
    }
    fn push_ptr(&self) {
        self.ptr_stk.push(*self.ptr);
    }
    fn pop_ptr(&self) {
        *self.ptr = self.ptr_stk.pop();
    }
}

impl ReprVisitor {
    // Various helpers for the TyVisitor impl

    #[inline]
    pub fn get<T>(&self, f: &fn(&T)) -> bool {
        unsafe {
            f(transmute::<*c_void,&T>(*self.ptr));
        }
        true
    }

    #[inline]
    pub fn visit_inner(&self, inner: *TyDesc) -> bool {
        self.visit_ptr_inner(*self.ptr, inner)
    }

    #[inline]
    pub fn visit_ptr_inner(&self, ptr: *c_void, inner: *TyDesc) -> bool {
        unsafe {
            let u = ReprVisitor(ptr, self.writer);
            let v = reflect::MovePtrAdaptor(u);
            visit_tydesc(inner, &v as &TyVisitor);
            true
        }
    }

    #[inline]
    pub fn write<T:Repr>(&self) -> bool {
        do self.get |v:&T| {
            v.write_repr(self.writer);
        }
    }

    pub fn write_escaped_slice(&self, slice: &str) {
        self.writer.write_char('"');
        for ch in slice.iter() {
            self.writer.write_escaped_char(ch);
        }
        self.writer.write_char('"');
    }

    pub fn write_mut_qualifier(&self, mtbl: uint) {
        if mtbl == 0 {
            self.writer.write_str("mut ");
        } else if mtbl == 1 {
            // skip, this is ast::m_imm
        } else {
            assert_eq!(mtbl, 2);
            self.writer.write_str("const ");
        }
    }

    pub fn write_vec_range(&self,
                           _mtbl: uint,
                           ptr: *(),
                           len: uint,
                           inner: *TyDesc)
                           -> bool {
        let mut p = ptr as *u8;
        let (sz, al) = unsafe { ((*inner).size, (*inner).align) };
        self.writer.write_char('[');
        let mut first = true;
        let mut left = len;
        // unit structs have 0 size, and don't loop forever.
        let dec = if sz == 0 {1} else {sz};
        while left > 0 {
            if first {
                first = false;
            } else {
                self.writer.write_str(", ");
            }
            self.visit_ptr_inner(p as *c_void, inner);
            p = align(ptr::offset(p, sz as int) as uint, al) as *u8;
            left -= dec;
        }
        self.writer.write_char(']');
        true
    }

    pub fn write_unboxed_vec_repr(&self,
                                  mtbl: uint,
                                  v: &raw::Vec<()>,
                                  inner: *TyDesc)
                                  -> bool {
        self.write_vec_range(mtbl, ptr::to_unsafe_ptr(&v.data),
                             v.fill, inner)
    }
}

impl TyVisitor for ReprVisitor {
    fn visit_bot(&self) -> bool {
        self.writer.write_str("!");
        true
    }
    fn visit_nil(&self) -> bool { self.write::<()>() }
    fn visit_bool(&self) -> bool { self.write::<bool>() }
    fn visit_int(&self) -> bool { self.write::<int>() }
    fn visit_i8(&self) -> bool { self.write::<i8>() }
    fn visit_i16(&self) -> bool { self.write::<i16>() }
    fn visit_i32(&self) -> bool { self.write::<i32>()  }
    fn visit_i64(&self) -> bool { self.write::<i64>() }

    fn visit_uint(&self) -> bool { self.write::<uint>() }
    fn visit_u8(&self) -> bool { self.write::<u8>() }
    fn visit_u16(&self) -> bool { self.write::<u16>() }
    fn visit_u32(&self) -> bool { self.write::<u32>() }
    fn visit_u64(&self) -> bool { self.write::<u64>() }

    fn visit_float(&self) -> bool { self.write::<float>() }
    fn visit_f32(&self) -> bool { self.write::<f32>() }
    fn visit_f64(&self) -> bool { self.write::<f64>() }

    fn visit_char(&self) -> bool {
        do self.get::<char> |&ch| {
            self.writer.write_char('\'');
            self.writer.write_escaped_char(ch);
            self.writer.write_char('\'');
        }
    }

    fn visit_estr_box(&self) -> bool {
        do self.get::<@str> |s| {
            self.writer.write_char('@');
            self.write_escaped_slice(*s);
        }
    }
    fn visit_estr_uniq(&self) -> bool {
        do self.get::<~str> |s| {
            self.writer.write_char('~');
            self.write_escaped_slice(*s);
        }
    }
    fn visit_estr_slice(&self) -> bool {
        do self.get::<&str> |s| {
            self.write_escaped_slice(*s);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_estr_fixed(&self, _n: uint, _sz: uint,
                        _align: uint) -> bool { fail!(); }

    fn visit_box(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('@');
        self.write_mut_qualifier(mtbl);
        do self.get::<&raw::Box<()>> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, inner);
        }
    }

    fn visit_uniq(&self, _mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('~');
        do self.get::<*c_void> |b| {
            self.visit_ptr_inner(*b, inner);
        }
    }

    fn visit_uniq_managed(&self, _mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('~');
        do self.get::<&raw::Box<()>> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, inner);
        }
    }

    fn visit_ptr(&self, _mtbl: uint, _inner: *TyDesc) -> bool {
        do self.get::<*c_void> |p| {
            self.writer.write_str(fmt!("(0x%x as *())",
                                       *p as uint));
        }
    }

    fn visit_rptr(&self, mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('&');
        self.write_mut_qualifier(mtbl);
        do self.get::<*c_void> |p| {
            self.visit_ptr_inner(*p, inner);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_vec(&self, _mtbl: uint, _inner: *TyDesc) -> bool { fail!(); }


    fn visit_unboxed_vec(&self, mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<raw::Vec<()>> |b| {
            self.write_unboxed_vec_repr(mtbl, b, inner);
        }
    }

    fn visit_evec_box(&self, mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<&raw::Box<raw::Vec<()>>> |b| {
            self.writer.write_char('@');
            self.write_mut_qualifier(mtbl);
            self.write_unboxed_vec_repr(mtbl, &b.data, inner);
        }
    }

    fn visit_evec_uniq(&self, mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<&raw::Vec<()>> |b| {
            self.writer.write_char('~');
            self.write_unboxed_vec_repr(mtbl, *b, inner);
        }
    }

    fn visit_evec_uniq_managed(&self, mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<&raw::Box<raw::Vec<()>>> |b| {
            self.writer.write_char('~');
            self.write_unboxed_vec_repr(mtbl, &b.data, inner);
        }
    }

    fn visit_evec_slice(&self, mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<raw::Slice<()>> |s| {
            self.writer.write_char('&');
            self.write_vec_range(mtbl, s.data, s.len, inner);
        }
    }

    fn visit_evec_fixed(&self, _n: uint, sz: uint, _align: uint,
                        mtbl: uint, inner: *TyDesc) -> bool {
        do self.get::<()> |b| {
            self.write_vec_range(mtbl, ptr::to_unsafe_ptr(b), sz, inner);
        }
    }

    fn visit_enter_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('{');
        true
    }

    fn visit_rec_field(&self, i: uint, name: &str,
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

    fn visit_leave_rec(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('}');
        true
    }

    fn visit_enter_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        self.writer.write_char('{');
        true
    }
    fn visit_class_field(&self, i: uint, name: &str,
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
    fn visit_leave_class(&self, _n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        self.writer.write_char('}');
        true
    }

    fn visit_enter_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.writer.write_char('(');
        true
    }
    fn visit_tup_field(&self, i: uint, inner: *TyDesc) -> bool {
        if i != 0 {
            self.writer.write_str(", ");
        }
        self.visit_inner(inner);
        true
    }
    fn visit_leave_tup(&self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        if _n_fields == 1 {
            self.writer.write_char(',');
        }
        self.writer.write_char(')');
        true
    }

    fn visit_enter_enum(&self,
                        _n_variants: uint,
                        get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint,
                        _align: uint) -> bool {
        let var_stk: &mut ~[VariantState] = self.var_stk;
        let disr = unsafe {
            get_disr(transmute(*self.ptr))
        };
        var_stk.push(SearchingFor(disr));
        true
    }

    fn visit_enter_enum_variant(&self, _variant: uint,
                                disr_val: int,
                                n_fields: uint,
                                name: &str) -> bool {
        let mut write = false;
        match self.var_stk.pop() {
            SearchingFor(sought) => {
                if disr_val == sought {
                    self.var_stk.push(Matched);
                    write = true;
                } else {
                    self.var_stk.push(SearchingFor(sought));
                }
            }
            Matched | AlreadyFound => {
                self.var_stk.push(AlreadyFound);
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

    fn visit_enum_variant_field(&self,
                                i: uint,
                                _offset: uint,
                                inner: *TyDesc)
                                -> bool {
        match self.var_stk[self.var_stk.len() - 1] {
            Matched => {
                if i != 0 {
                    self.writer.write_str(", ");
                }
                if ! self.visit_inner(inner) {
                    return false;
                }
            }
            _ => ()
        }
        true
    }

    fn visit_leave_enum_variant(&self, _variant: uint,
                                _disr_val: int,
                                n_fields: uint,
                                _name: &str) -> bool {
        match self.var_stk[self.var_stk.len() - 1] {
            Matched => {
                if n_fields > 0 {
                    self.writer.write_char(')');
                }
            }
            _ => ()
        }
        true
    }

    fn visit_leave_enum(&self,
                        _n_variants: uint,
                        _get_disr: extern unsafe fn(ptr: *Opaque) -> int,
                        _sz: uint,
                        _align: uint)
                        -> bool {
        let var_stk: &mut ~[VariantState] = self.var_stk;
        match var_stk.pop() {
            SearchingFor(*) => fail!("enum value matched no variant"),
            _ => true
        }
    }

    fn visit_enter_fn(&self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(&self, _i: uint, _mode: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_fn_output(&self, _retstyle: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_leave_fn(&self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait(&self) -> bool { true }
    fn visit_var(&self) -> bool { true }
    fn visit_var_integral(&self) -> bool { true }
    fn visit_param(&self, _i: uint) -> bool { true }
    fn visit_self(&self) -> bool { true }
    fn visit_type(&self) -> bool { true }

    fn visit_opaque_box(&self) -> bool {
        self.writer.write_char('@');
        do self.get::<&raw::Box<()>> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, b.type_desc);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_constr(&self, _inner: *TyDesc) -> bool { fail!(); }

    fn visit_closure_ptr(&self, _ck: uint) -> bool { true }
}

pub fn write_repr<T>(writer: @Writer, object: &T) {
    unsafe {
        let ptr = ptr::to_unsafe_ptr(object) as *c_void;
        let tydesc = get_tydesc::<T>();
        let u = ReprVisitor(ptr, writer);
        let v = reflect::MovePtrAdaptor(u);
        visit_tydesc(tydesc, &v as &TyVisitor)
    }
}

#[cfg(test)]
struct P {a: int, b: float}

#[test]
fn test_repr() {

    fn exact_test<T>(t: &T, e:&str) {
        let s : &str = io::with_str_writer(|w| write_repr(w, t));
        if s != e {
            error!("expected '%s', got '%s'",
                   e, s);
        }
        assert_eq!(s, e);
    }

    exact_test(&10, "10");
    exact_test(&true, "true");
    exact_test(&false, "false");
    exact_test(&1.234, "1.234");
    exact_test(&(&"hello"), "\"hello\"");
    exact_test(&(@"hello"), "@\"hello\"");
    exact_test(&(~"he\u10f3llo"), "~\"he\\u10f3llo\"");

    exact_test(&(@10), "@10");
    exact_test(&(@mut 10), "@10"); // FIXME: #4210: incorrect
    exact_test(&((@mut 10, 2)), "(@mut 10, 2)");
    exact_test(&(~10), "~10");
    exact_test(&(&10), "&10");
    let mut x = 10;
    exact_test(&(&mut x), "&mut 10");
    exact_test(&(@mut [1, 2]), "@mut [1, 2]");

    exact_test(&(1,), "(1,)");
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
    exact_test(&(P{a:10, b:1.234}),
               "{a: 10, b: 1.234}");
    exact_test(&(@P{a:10, b:1.234}),
               "@{a: 10, b: 1.234}");
    exact_test(&(~P{a:10, b:1.234}),
               "~{a: 10, b: 1.234}");
    exact_test(&(10_u8, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u16, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u32, ~"hello"),
               "(10, ~\"hello\")");
    exact_test(&(10_u64, ~"hello"),
               "(10, ~\"hello\")");

    struct Foo;
    exact_test(&(~[Foo, Foo, Foo]), "~[{}, {}, {}]");
}
