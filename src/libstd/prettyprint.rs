// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use serialize;

use core::io::WriterUtil;
use core::io;

pub struct Serializer {
    wr: @io::Writer,
}

pub fn Serializer(wr: @io::Writer) -> Serializer {
    Serializer { wr: wr }
}

impl serialize::Encoder for Serializer {
    fn emit_nil(&self) {
        self.wr.write_str(~"()")
    }

    fn emit_uint(&self, v: uint) {
        self.wr.write_str(fmt!("%?u", v));
    }

    fn emit_u64(&self, v: u64) {
        self.wr.write_str(fmt!("%?_u64", v));
    }

    fn emit_u32(&self, v: u32) {
        self.wr.write_str(fmt!("%?_u32", v));
    }

    fn emit_u16(&self, v: u16) {
        self.wr.write_str(fmt!("%?_u16", v));
    }

    fn emit_u8(&self, v: u8) {
        self.wr.write_str(fmt!("%?_u8", v));
    }

    fn emit_int(&self, v: int) {
        self.wr.write_str(fmt!("%?", v));
    }

    fn emit_i64(&self, v: i64) {
        self.wr.write_str(fmt!("%?_i64", v));
    }

    fn emit_i32(&self, v: i32) {
        self.wr.write_str(fmt!("%?_i32", v));
    }

    fn emit_i16(&self, v: i16) {
        self.wr.write_str(fmt!("%?_i16", v));
    }

    fn emit_i8(&self, v: i8) {
        self.wr.write_str(fmt!("%?_i8", v));
    }

    fn emit_bool(&self, v: bool) {
        self.wr.write_str(fmt!("%b", v));
    }

    fn emit_float(&self, v: float) {
        self.wr.write_str(fmt!("%?_f", v));
    }

    fn emit_f64(&self, v: f64) {
        self.wr.write_str(fmt!("%?_f64", v));
    }

    fn emit_f32(&self, v: f32) {
        self.wr.write_str(fmt!("%?_f32", v));
    }

    fn emit_char(&self, v: char) {
        self.wr.write_str(fmt!("%?", v));
    }

    fn emit_borrowed_str(&self, v: &str) {
        self.wr.write_str(fmt!("&%?", v));
    }

    fn emit_owned_str(&self, v: &str) {
        self.wr.write_str(fmt!("~%?", v));
    }

    fn emit_managed_str(&self, v: &str) {
        self.wr.write_str(fmt!("@%?", v));
    }

    fn emit_borrowed(&self, f: &fn()) {
        self.wr.write_str(~"&");
        f();
    }

    fn emit_owned(&self, f: &fn()) {
        self.wr.write_str(~"~");
        f();
    }

    fn emit_managed(&self, f: &fn()) {
        self.wr.write_str(~"@");
        f();
    }

    fn emit_enum(&self, _name: &str, f: &fn()) {
        f();
    }

    fn emit_enum_variant(&self, v_name: &str, _v_id: uint, sz: uint,
                         f: &fn()) {
        self.wr.write_str(v_name);
        if sz > 0u { self.wr.write_str(~"("); }
        f();
        if sz > 0u { self.wr.write_str(~")"); }
    }

    fn emit_enum_variant_arg(&self, idx: uint, f: &fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }

    fn emit_borrowed_vec(&self, _len: uint, f: &fn()) {
        self.wr.write_str(~"&[");
        f();
        self.wr.write_str(~"]");
    }

    fn emit_owned_vec(&self, _len: uint, f: &fn()) {
        self.wr.write_str(~"~[");
        f();
        self.wr.write_str(~"]");
    }

    fn emit_managed_vec(&self, _len: uint, f: &fn()) {
        self.wr.write_str(~"@[");
        f();
        self.wr.write_str(~"]");
    }

    fn emit_vec_elt(&self, idx: uint, f: &fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }

    fn emit_rec(&self, f: &fn()) {
        self.wr.write_str(~"{");
        f();
        self.wr.write_str(~"}");
    }

    fn emit_struct(&self, name: &str, _len: uint, f: &fn()) {
        self.wr.write_str(fmt!("%s {", name));
        f();
        self.wr.write_str(~"}");
    }

    fn emit_field(&self, name: &str, idx: uint, f: &fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        self.wr.write_str(name);
        self.wr.write_str(~": ");
        f();
    }

    fn emit_tup(&self, _len: uint, f: &fn()) {
        self.wr.write_str(~"(");
        f();
        self.wr.write_str(~")");
    }

    fn emit_tup_elt(&self, idx: uint, f: &fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }

    fn emit_option(&self, f: &fn()) {
        f();
    }

    fn emit_option_none(&self) {
        self.wr.write_str("None");
    }

    fn emit_option_some(&self, f: &fn()) {
        self.wr.write_str("Some(");
        f();
        self.wr.write_char(')');
    }
}
