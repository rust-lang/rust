// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
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

use std::char;
use std::intrinsics::{Disr, Opaque, TyDesc, TyVisitor, get_tydesc, visit_tydesc};
use std::io;
use std::mem;
use std::raw;

use reflect;
use reflect::{MovePtr, align};

macro_rules! try( ($me:expr, $e:expr) => (
    match $e {
        Ok(()) => {},
        Err(e) => { $me.last_err = Some(e); return false; }
    }
) )

/// Representations

pub trait Repr {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()>;
}

impl Repr for () {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        writer.write("()".as_bytes())
    }
}

impl Repr for bool {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        let s = if *self { "true" } else { "false" };
        writer.write(s.as_bytes())
    }
}

impl Repr for int {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        write!(writer, "{}", *self)
    }
}

macro_rules! int_repr(($ty:ident, $suffix:expr) => (impl Repr for $ty {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        write!(writer, "{}{}", *self, $suffix)
    }
}))

int_repr!(i8, "i8")
int_repr!(i16, "i16")
int_repr!(i32, "i32")
int_repr!(i64, "i64")
int_repr!(uint, "u")
int_repr!(u8, "u8")
int_repr!(u16, "u16")
int_repr!(u32, "u32")
int_repr!(u64, "u64")

macro_rules! num_repr(($ty:ident, $suffix:expr) => (impl Repr for $ty {
    fn write_repr(&self, writer: &mut io::Writer) -> io::IoResult<()> {
        let s = self.to_string();
        writer.write(s.as_bytes()).and_then(|()| {
            writer.write($suffix)
        })
    }
}))

num_repr!(f32, b"f32")
num_repr!(f64, b"f64")

// New implementation using reflect::MovePtr

enum VariantState {
    SearchingFor(Disr),
    Matched,
    AlreadyFound
}

pub struct ReprVisitor<'a> {
    ptr: *const u8,
    ptr_stk: Vec<*const u8>,
    var_stk: Vec<VariantState>,
    writer: &'a mut io::Writer+'a,
    last_err: Option<io::IoError>,
}

impl<'a> MovePtr for ReprVisitor<'a> {
    #[inline]
    fn move_ptr(&mut self, adjustment: |*const u8| -> *const u8) {
        self.ptr = adjustment(self.ptr);
    }
    fn push_ptr(&mut self) {
        self.ptr_stk.push(self.ptr);
    }
    fn pop_ptr(&mut self) {
        self.ptr = self.ptr_stk.pop().unwrap();
    }
}

impl<'a> ReprVisitor<'a> {
    // Various helpers for the TyVisitor impl
    pub fn new(ptr: *const u8, writer: &'a mut io::Writer) -> ReprVisitor<'a> {
        ReprVisitor {
            ptr: ptr,
            ptr_stk: vec!(),
            var_stk: vec!(),
            writer: writer,
            last_err: None,
        }
    }

    #[inline]
    pub fn get<T>(&mut self, f: |&mut ReprVisitor, &T| -> bool) -> bool {
        unsafe {
            let ptr = self.ptr;
            f(self, mem::transmute::<*const u8,&T>(ptr))
        }
    }

    #[inline]
    pub fn visit_inner(&mut self, inner: *const TyDesc) -> bool {
        let ptr = self.ptr;
        self.visit_ptr_inner(ptr, inner)
    }

    #[inline]
    pub fn visit_ptr_inner(&mut self, ptr: *const u8,
                           inner: *const TyDesc) -> bool {
        unsafe {
            let u = ReprVisitor::new(ptr, mem::transmute_copy(&self.writer));
            let mut v = reflect::MovePtrAdaptor::new(u);
            // Obviously this should not be a thing, but blame #8401 for now
            visit_tydesc(inner, &mut v as &mut TyVisitor);
            match v.unwrap().last_err {
                Some(e) => {
                    self.last_err = Some(e);
                    false
                }
                None => true,
            }
        }
    }

    #[inline]
    pub fn write<T:Repr>(&mut self) -> bool {
        self.get(|this, v:&T| {
            try!(this, v.write_repr(this.writer));
            true
        })
    }

    pub fn write_escaped_slice(&mut self, slice: &str) -> bool {
        try!(self, self.writer.write([b'"']));
        for ch in slice.chars() {
            if !self.write_escaped_char(ch, true) { return false }
        }
        try!(self, self.writer.write([b'"']));
        true
    }

    pub fn write_mut_qualifier(&mut self, mtbl: uint) -> bool {
        if mtbl == 0 {
            try!(self, self.writer.write("mut ".as_bytes()));
        } else if mtbl == 1 {
            // skip, this is ast::m_imm
        } else {
            fail!("invalid mutability value");
        }
        true
    }

    pub fn write_vec_range(&mut self, ptr: *const (), len: uint,
                           inner: *const TyDesc) -> bool {
        let mut p = ptr as *const u8;
        let (sz, al) = unsafe { ((*inner).size, (*inner).align) };
        try!(self, self.writer.write([b'[']));
        let mut first = true;
        let mut left = len;
        // unit structs have 0 size, and don't loop forever.
        let dec = if sz == 0 {1} else {sz};
        while left > 0 {
            if first {
                first = false;
            } else {
                try!(self, self.writer.write(", ".as_bytes()));
            }
            self.visit_ptr_inner(p as *const u8, inner);
            p = align(unsafe { p.offset(sz as int) as uint }, al) as *const u8;
            left -= dec;
        }
        try!(self, self.writer.write([b']']));
        true
    }

    fn write_escaped_char(&mut self, ch: char, is_str: bool) -> bool {
        try!(self, match ch {
            '\t' => self.writer.write("\\t".as_bytes()),
            '\r' => self.writer.write("\\r".as_bytes()),
            '\n' => self.writer.write("\\n".as_bytes()),
            '\\' => self.writer.write("\\\\".as_bytes()),
            '\'' => {
                if is_str {
                    self.writer.write("'".as_bytes())
                } else {
                    self.writer.write("\\'".as_bytes())
                }
            }
            '"' => {
                if is_str {
                    self.writer.write("\\\"".as_bytes())
                } else {
                    self.writer.write("\"".as_bytes())
                }
            }
            '\x20'...'\x7e' => self.writer.write([ch as u8]),
            _ => {
                char::escape_unicode(ch, |c| {
                    let _ = self.writer.write([c as u8]);
                });
                Ok(())
            }
        });
        return true;
    }
}

impl<'a> TyVisitor for ReprVisitor<'a> {
    fn visit_bot(&mut self) -> bool {
        try!(self, self.writer.write("!".as_bytes()));
        true
    }
    fn visit_nil(&mut self) -> bool { self.write::<()>() }
    fn visit_bool(&mut self) -> bool { self.write::<bool>() }
    fn visit_int(&mut self) -> bool { self.write::<int>() }
    fn visit_i8(&mut self) -> bool { self.write::<i8>() }
    fn visit_i16(&mut self) -> bool { self.write::<i16>() }
    fn visit_i32(&mut self) -> bool { self.write::<i32>()  }
    fn visit_i64(&mut self) -> bool { self.write::<i64>() }

    fn visit_uint(&mut self) -> bool { self.write::<uint>() }
    fn visit_u8(&mut self) -> bool { self.write::<u8>() }
    fn visit_u16(&mut self) -> bool { self.write::<u16>() }
    fn visit_u32(&mut self) -> bool { self.write::<u32>() }
    fn visit_u64(&mut self) -> bool { self.write::<u64>() }

    fn visit_f32(&mut self) -> bool { self.write::<f32>() }
    fn visit_f64(&mut self) -> bool { self.write::<f64>() }

    fn visit_char(&mut self) -> bool {
        self.get::<char>(|this, &ch| {
            try!(this, this.writer.write([b'\'']));
            if !this.write_escaped_char(ch, false) { return false }
            try!(this, this.writer.write([b'\'']));
            true
        })
    }

    fn visit_estr_slice(&mut self) -> bool {
        self.get::<&str>(|this, s| this.write_escaped_slice(*s))
    }

    fn visit_box(&mut self, _mtbl: uint, _inner: *const TyDesc) -> bool {
        try!(self, self.writer.write("box(GC) ???".as_bytes()));
        true
    }

    fn visit_uniq(&mut self, _mtbl: uint, inner: *const TyDesc) -> bool {
        try!(self, self.writer.write("box ".as_bytes()));
        self.get::<*const u8>(|this, b| {
            this.visit_ptr_inner(*b, inner)
        })
    }

    fn visit_ptr(&mut self, mtbl: uint, _inner: *const TyDesc) -> bool {
        self.get::<*const u8>(|this, p| {
            try!(this, write!(this.writer, "({} as *", *p));
            if mtbl == 0 {
                try!(this, this.writer.write("mut ".as_bytes()));
            } else if mtbl == 1 {
                try!(this, this.writer.write("const ".as_bytes()));
            } else {
                fail!("invalid mutability value");
            }
            try!(this, this.writer.write("())".as_bytes()));
            true
        })
    }

    fn visit_rptr(&mut self, mtbl: uint, inner: *const TyDesc) -> bool {
        try!(self, self.writer.write([b'&']));
        self.write_mut_qualifier(mtbl);
        self.get::<*const u8>(|this, p| {
            this.visit_ptr_inner(*p, inner)
        })
    }

    fn visit_evec_slice(&mut self, mtbl: uint, inner: *const TyDesc) -> bool {
        self.get::<raw::Slice<()>>(|this, s| {
            try!(this, this.writer.write([b'&']));
            this.write_mut_qualifier(mtbl);
            let size = unsafe {
                if (*inner).size == 0 { 1 } else { (*inner).size }
            };
            this.write_vec_range(s.data, s.len * size, inner)
        })
    }

    fn visit_evec_fixed(&mut self, n: uint, sz: uint, _align: uint,
                        inner: *const TyDesc) -> bool {
        let assumed_size = if sz == 0 { n } else { sz };
        self.get::<()>(|this, b| {
            this.write_vec_range(b, assumed_size, inner)
        })
    }


    fn visit_enter_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        try!(self, self.writer.write([b'{']));
        true
    }

    fn visit_rec_field(&mut self, i: uint, name: &str,
                       mtbl: uint, inner: *const TyDesc) -> bool {
        if i != 0 {
            try!(self, self.writer.write(", ".as_bytes()));
        }
        self.write_mut_qualifier(mtbl);
        try!(self, self.writer.write(name.as_bytes()));
        try!(self, self.writer.write(": ".as_bytes()));
        self.visit_inner(inner);
        true
    }

    fn visit_leave_rec(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        try!(self, self.writer.write([b'}']));
        true
    }

    fn visit_enter_class(&mut self, name: &str, named_fields: bool, n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        try!(self, self.writer.write(name.as_bytes()));
        if n_fields != 0 {
            if named_fields {
                try!(self, self.writer.write([b'{']));
            } else {
                try!(self, self.writer.write([b'(']));
            }
        }
        true
    }

    fn visit_class_field(&mut self, i: uint, name: &str, named: bool,
                         _mtbl: uint, inner: *const TyDesc) -> bool {
        if i != 0 {
            try!(self, self.writer.write(", ".as_bytes()));
        }
        if named {
            try!(self, self.writer.write(name.as_bytes()));
            try!(self, self.writer.write(": ".as_bytes()));
        }
        self.visit_inner(inner);
        true
    }

    fn visit_leave_class(&mut self, _name: &str, named_fields: bool, n_fields: uint,
                         _sz: uint, _align: uint) -> bool {
        if n_fields != 0 {
            if named_fields {
                try!(self, self.writer.write([b'}']));
            } else {
                try!(self, self.writer.write([b')']));
            }
        }
        true
    }

    fn visit_enter_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        try!(self, self.writer.write([b'(']));
        true
    }

    fn visit_tup_field(&mut self, i: uint, inner: *const TyDesc) -> bool {
        if i != 0 {
            try!(self, self.writer.write(", ".as_bytes()));
        }
        self.visit_inner(inner);
        true
    }

    fn visit_leave_tup(&mut self, _n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        if _n_fields == 1 {
            try!(self, self.writer.write([b',']));
        }
        try!(self, self.writer.write([b')']));
        true
    }

    fn visit_enter_enum(&mut self,
                        _n_variants: uint,
                        get_disr: unsafe extern fn(ptr: *const Opaque) -> Disr,
                        _sz: uint,
                        _align: uint) -> bool {
        let disr = unsafe {
            get_disr(mem::transmute(self.ptr))
        };
        self.var_stk.push(SearchingFor(disr));
        true
    }

    fn visit_enter_enum_variant(&mut self, _variant: uint,
                                disr_val: Disr,
                                n_fields: uint,
                                name: &str) -> bool {
        let mut write = false;
        match self.var_stk.pop().unwrap() {
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
            try!(self, self.writer.write(name.as_bytes()));
            if n_fields > 0 {
                try!(self, self.writer.write([b'(']));
            }
        }
        true
    }

    fn visit_enum_variant_field(&mut self,
                                i: uint,
                                _offset: uint,
                                inner: *const TyDesc)
                                -> bool {
        match self.var_stk[self.var_stk.len() - 1] {
            Matched => {
                if i != 0 {
                    try!(self, self.writer.write(", ".as_bytes()));
                }
                if ! self.visit_inner(inner) {
                    return false;
                }
            }
            _ => ()
        }
        true
    }

    fn visit_leave_enum_variant(&mut self, _variant: uint,
                                _disr_val: Disr,
                                n_fields: uint,
                                _name: &str) -> bool {
        match self.var_stk[self.var_stk.len() - 1] {
            Matched => {
                if n_fields > 0 {
                    try!(self, self.writer.write([b')']));
                }
            }
            _ => ()
        }
        true
    }

    fn visit_leave_enum(&mut self,
                        _n_variants: uint,
                        _get_disr: unsafe extern fn(ptr: *const Opaque) -> Disr,
                        _sz: uint,
                        _align: uint)
                        -> bool {
        match self.var_stk.pop().unwrap() {
            SearchingFor(..) => fail!("enum value matched no variant"),
            _ => true
        }
    }

    fn visit_enter_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool {
        try!(self, self.writer.write("fn(".as_bytes()));
        true
    }

    fn visit_fn_input(&mut self, i: uint, _mode: uint,
                      inner: *const TyDesc) -> bool {
        if i != 0 {
            try!(self, self.writer.write(", ".as_bytes()));
        }
        let name = unsafe { (*inner).name };
        try!(self, self.writer.write(name.as_bytes()));
        true
    }

    fn visit_fn_output(&mut self, _retstyle: uint, variadic: bool,
                       inner: *const TyDesc) -> bool {
        if variadic {
            try!(self, self.writer.write(", ...".as_bytes()));
        }
        try!(self, self.writer.write(")".as_bytes()));
        let name = unsafe { (*inner).name };
        if name != "()" {
            try!(self, self.writer.write(" -> ".as_bytes()));
            try!(self, self.writer.write(name.as_bytes()));
        }
        true
    }

    fn visit_leave_fn(&mut self, _purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait(&mut self, name: &str) -> bool {
        try!(self, self.writer.write(name.as_bytes()));
        true
    }

    fn visit_param(&mut self, _i: uint) -> bool { true }
    fn visit_self(&mut self) -> bool { true }
}

pub fn write_repr<T>(writer: &mut io::Writer, object: &T) -> io::IoResult<()> {
    unsafe {
        let ptr = object as *const T as *const u8;
        let tydesc = get_tydesc::<T>();
        let u = ReprVisitor::new(ptr, writer);
        let mut v = reflect::MovePtrAdaptor::new(u);
        visit_tydesc(tydesc, &mut v as &mut TyVisitor);
        match v.unwrap().last_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

pub fn repr_to_string<T>(t: &T) -> String {
    let mut result = io::MemWriter::new();
    write_repr(&mut result as &mut io::Writer, t).unwrap();
    String::from_utf8(result.unwrap()).unwrap()
}

#[cfg(test)]
#[allow(dead_code)]
struct P {a: int, b: f64}

#[test]
fn test_repr() {
    use std::io::stdio::println;
    use std::char::is_alphabetic;
    use std::mem::swap;

    fn exact_test<T>(t: &T, e:&str) {
        let mut m = io::MemWriter::new();
        write_repr(&mut m as &mut io::Writer, t).unwrap();
        let s = String::from_utf8(m.unwrap()).unwrap();
        assert_eq!(s.as_slice(), e);
    }

    exact_test(&10i, "10");
    exact_test(&true, "true");
    exact_test(&false, "false");
    exact_test(&1.234f64, "1.234f64");
    exact_test(&("hello"), "\"hello\"");

    exact_test(&(box 10i), "box 10");
    exact_test(&(&10i), "&10");
    let mut x = 10i;
    exact_test(&(&mut x), "&mut 10");

    exact_test(&(0i as *const()), "(0x0 as *const ())");
    exact_test(&(0i as *mut ()), "(0x0 as *mut ())");

    exact_test(&(1i,), "(1,)");
    exact_test(&(&["hi", "there"]),
               "&[\"hi\", \"there\"]");
    exact_test(&(P{a:10, b:1.234}),
               "repr::P{a: 10, b: 1.234f64}");
    exact_test(&(box P{a:10, b:1.234}),
               "box repr::P{a: 10, b: 1.234f64}");

    exact_test(&(&[1i, 2i]), "&[1, 2]");
    exact_test(&(&mut [1i, 2i]), "&mut [1, 2]");

    exact_test(&'\'', "'\\''");
    exact_test(&'"', "'\"'");
    exact_test(&("'"), "\"'\"");
    exact_test(&("\""), "\"\\\"\"");

    exact_test(&println, "fn(&str)");
    exact_test(&swap::<int>, "fn(&mut int, &mut int)");
    exact_test(&is_alphabetic, "fn(char) -> bool");

    struct Bar(int, int);
    exact_test(&(Bar(2, 2)), "repr::test_repr::Bar(2, 2)");
}
