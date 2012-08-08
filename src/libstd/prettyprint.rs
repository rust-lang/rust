import io::writer;
import io::writer_util;
import serialization::serializer;

impl writer: serializer {
    fn emit_nil() {
        self.write_str(~"()")
    }

    fn emit_uint(v: uint) {
        self.write_str(fmt!{"%?u", v});
    }

    fn emit_u64(v: u64) {
        self.write_str(fmt!{"%?_u64", v});
    }

    fn emit_u32(v: u32) {
        self.write_str(fmt!{"%?_u32", v});
    }

    fn emit_u16(v: u16) {
        self.write_str(fmt!{"%?_u16", v});
    }

    fn emit_u8(v: u8) {
        self.write_str(fmt!{"%?_u8", v});
    }

    fn emit_int(v: int) {
        self.write_str(fmt!{"%?", v});
    }

    fn emit_i64(v: i64) {
        self.write_str(fmt!{"%?_i64", v});
    }

    fn emit_i32(v: i32) {
        self.write_str(fmt!{"%?_i32", v});
    }

    fn emit_i16(v: i16) {
        self.write_str(fmt!{"%?_i16", v});
    }

    fn emit_i8(v: i8) {
        self.write_str(fmt!{"%?_i8", v});
    }

    fn emit_bool(v: bool) {
        self.write_str(fmt!{"%b", v});
    }

    fn emit_float(v: float) {
        self.write_str(fmt!{"%?_f", v});
    }

    fn emit_f64(v: f64) {
        self.write_str(fmt!{"%?_f64", v});
    }

    fn emit_f32(v: f32) {
        self.write_str(fmt!{"%?_f32", v});
    }

    fn emit_str(v: ~str) {
        self.write_str(fmt!{"%?", v});
    }

    fn emit_enum(_name: ~str, f: fn()) {
        f();
    }

    fn emit_enum_variant(v_name: ~str, _v_id: uint, sz: uint, f: fn()) {
        self.write_str(v_name);
        if sz > 0u { self.write_str(~"("); }
        f();
        if sz > 0u { self.write_str(~")"); }
    }

    fn emit_enum_variant_arg(idx: uint, f: fn()) {
        if idx > 0u { self.write_str(~", "); }
        f();
    }

    fn emit_vec(_len: uint, f: fn()) {
        self.write_str(~"[");
        f();
        self.write_str(~"]");
    }

    fn emit_vec_elt(idx: uint, f: fn()) {
        if idx > 0u { self.write_str(~", "); }
        f();
    }

    fn emit_box(f: fn()) {
        self.write_str(~"@");
        f();
    }

    fn emit_uniq(f: fn()) {
        self.write_str(~"~");
        f();
    }

    fn emit_rec(f: fn()) {
        self.write_str(~"{");
        f();
        self.write_str(~"}");
    }

    fn emit_rec_field(f_name: ~str, f_idx: uint, f: fn()) {
        if f_idx > 0u { self.write_str(~", "); }
        self.write_str(f_name);
        self.write_str(~": ");
        f();
    }

    fn emit_tup(_sz: uint, f: fn()) {
        self.write_str(~"(");
        f();
        self.write_str(~")");
    }

    fn emit_tup_elt(idx: uint, f: fn()) {
        if idx > 0u { self.write_str(~", "); }
        f();
    }
}