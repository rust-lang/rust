#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use io::Writer;
use io::WriterUtil;
use serialization2;

struct Serializer {
    wr: io::Writer,
}

fn Serializer(wr: io::Writer) -> Serializer {
    Serializer { wr: wr }
}

impl Serializer: serialization2::Serializer {
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

    fn emit_str(&self, v: &str) {
        self.wr.write_str(fmt!("%?", v));
    }

    fn emit_enum(&self, _name: &str, f: fn()) {
        f();
    }

    fn emit_enum_variant(&self, v_name: &str, _v_id: uint, sz: uint,
                         f: fn()) {
        self.wr.write_str(v_name);
        if sz > 0u { self.wr.write_str(~"("); }
        f();
        if sz > 0u { self.wr.write_str(~")"); }
    }

    fn emit_enum_variant_arg(&self, idx: uint, f: fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }

    fn emit_vec(&self, _len: uint, f: fn()) {
        self.wr.write_str(~"[");
        f();
        self.wr.write_str(~"]");
    }

    fn emit_vec_elt(&self, idx: uint, f: fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }

    fn emit_box(&self, f: fn()) {
        self.wr.write_str(~"@");
        f();
    }

    fn emit_uniq(&self, f: fn()) {
        self.wr.write_str(~"~");
        f();
    }

    fn emit_rec(&self, f: fn()) {
        self.wr.write_str(~"{");
        f();
        self.wr.write_str(~"}");
    }

    fn emit_rec_field(&self, f_name: &str, f_idx: uint, f: fn()) {
        if f_idx > 0u { self.wr.write_str(~", "); }
        self.wr.write_str(f_name);
        self.wr.write_str(~": ");
        f();
    }

    fn emit_tup(&self, _sz: uint, f: fn()) {
        self.wr.write_str(~"(");
        f();
        self.wr.write_str(~")");
    }

    fn emit_tup_elt(&self, idx: uint, f: fn()) {
        if idx > 0u { self.wr.write_str(~", "); }
        f();
    }
}
