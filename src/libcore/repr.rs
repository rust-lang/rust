/*!

More runtime type reflection

*/

#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use dvec::DVec;
use io::{Writer, WriterUtil};
use libc::c_void;
use sys::TypeDesc;
use to_str::ToStr;
use cast::transmute;
use intrinsic::{TyDesc, TyVisitor, visit_tydesc};
use reflect::{MovePtr, MovePtrAdaptor};
use vec::UnboxedVecRepr;
use vec::raw::{VecRepr, SliceRepr};
pub use box::raw::BoxRepr;
use box::raw::BoxHeaderRepr;

/// Helpers

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1) & !(align - 1)
}

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
                // XXX: This is inefficient because it requires a malloc.
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
    // XXX: This can lose precision.
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
    // XXX: This can lose precision.
    fn write_repr(writer: @Writer) { writer.write_uint(self as uint); }
}

impl float : Repr {
    // XXX: This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}
impl f32 : Repr {
    // XXX: This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}
impl f64 : Repr {
    // XXX: This mallocs.
    fn write_repr(writer: @Writer) { writer.write_str(self.to_str()); }
}

impl char : Repr {
    fn write_repr(writer: @Writer) { writer.write_char(self); }
}


// New implementation using reflect::MovePtr

struct ReprVisitor {
    mut ptr: *c_void,
    writer: @Writer
}
fn ReprVisitor(ptr: *c_void, writer: @Writer) -> ReprVisitor {
    ReprVisitor { ptr: ptr, writer: writer }
}

impl ReprVisitor : MovePtr {
    #[inline(always)]
    fn move_ptr(adjustment: fn(*c_void) -> *c_void) {
        self.ptr = adjustment(self.ptr);
    }
}

impl ReprVisitor {

    // Various helpers for the TyVisitor impl

    #[inline(always)]
    fn get<T>(f: fn((&T))) -> bool {
        unsafe {
            f(transmute::<*c_void,&T>(copy self.ptr));
        }
        true
    }

    #[inline(always)]
    fn visit_inner(inner: *TyDesc) -> bool {
        self.visit_ptr_inner(self.ptr, inner)
    }

    #[inline(always)]
    fn visit_ptr_inner(ptr: *c_void, inner: *TyDesc) -> bool {
        let mut u = ReprVisitor(ptr, self.writer);
        let v = reflect::MovePtrAdaptor(move u);
        visit_tydesc(inner, (move v) as @TyVisitor);
        true
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

    fn visit_char() -> bool { self.write::<uint>() }

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
        do self.get::<&box::raw::BoxRepr> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, inner);
        }
    }

    fn visit_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.writer.write_char('~');
        self.write_mut_qualifier(mtbl);
        do self.get::<&box::raw::BoxRepr> |b| {
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

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        true
    }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool { true }
    fn visit_enum_variant_field(_i: uint, inner: *TyDesc) -> bool {
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
        do self.get::<&box::raw::BoxRepr> |b| {
            let p = ptr::to_unsafe_ptr(&b.data) as *c_void;
            self.visit_ptr_inner(p, b.header.type_desc);
        }
    }

    // Type no longer exists, vestigial function.
    fn visit_constr(_inner: *TyDesc) -> bool { fail; }

    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

pub fn write_repr2<T>(writer: @Writer, object: &T) {
    let ptr = ptr::to_unsafe_ptr(object) as *c_void;
    let tydesc = intrinsic::get_tydesc::<T>();
    let mut u = ReprVisitor(ptr, writer);
    let v = reflect::MovePtrAdaptor(move u);
    visit_tydesc(tydesc, (move v) as @TyVisitor)
}

#[test]
fn test_repr2() {
    repr::write_repr2(io::stdout(), &10);
    io::println("");
    repr::write_repr2(io::stdout(), &true);
    io::println("");
    repr::write_repr2(io::stdout(), &false);
    io::println("");
    repr::write_repr2(io::stdout(), &1.234);
    io::println("");
    repr::write_repr2(io::stdout(), &(&"hello"));
    io::println("");
    repr::write_repr2(io::stdout(), &(@"hello"));
    io::println("");
    repr::write_repr2(io::stdout(), &(~"he\u10f3llo"));
    io::println("");
    repr::write_repr2(io::stdout(), &(@10));
    io::println("");
    repr::write_repr2(io::stdout(), &(@mut 10));
    io::println("");
    repr::write_repr2(io::stdout(), &(~10));
    io::println("");
    repr::write_repr2(io::stdout(), &(~mut 10));
    io::println("");
    repr::write_repr2(io::stdout(), &(&10));
    io::println("");
    let mut x = 10;
    repr::write_repr2(io::stdout(), &(&mut x));
    io::println("");
    repr::write_repr2(io::stdout(), &(ptr::to_unsafe_ptr(&10) as *int));
    io::println("");
    repr::write_repr2(io::stdout(), &(@[1,2,3,4,5,6,7,8]));
    io::println("");
    repr::write_repr2(io::stdout(), &(@[1u8,2u8,3u8,4u8]));
    io::println("");
    repr::write_repr2(io::stdout(), &(@["hi", "there"]));
    io::println("");
    repr::write_repr2(io::stdout(), &(~["hi", "there"]));
    io::println("");
    repr::write_repr2(io::stdout(), &(&["hi", "there"]));
    io::println("");
    repr::write_repr2(io::stdout(), &({a:10, b:1.234}));
    io::println("");
    repr::write_repr2(io::stdout(), &(@{a:10, b:1.234}));
    io::println("");
    repr::write_repr2(io::stdout(), &(~{a:10, b:1.234}));
    io::println("");
}


// Old non-factored implementation, transitional...

enum EnumVisitState {
    PreVariant,     // We're before the variant we're interested in.
    InVariant,      // We're inside the variant we're interested in.
    PostVariant,    // We're after the variant we're interested in.
    Degenerate      // This is a degenerate enum (exactly 1 variant)
}

impl EnumVisitState : cmp::Eq {
    #[cfg(stage0)]
    pure fn eq(other: &EnumVisitState) -> bool {
        (self as uint) == ((*other) as uint)
    }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn eq(&self, other: &EnumVisitState) -> bool {
        ((*self) as uint) == ((*other) as uint)
    }
    #[cfg(stage0)]
    pure fn ne(other: &EnumVisitState) -> bool { !self.eq(other) }
    #[cfg(stage1)]
    #[cfg(stage2)]
    pure fn ne(&self, other: &EnumVisitState) -> bool { !(*self).eq(other) }
}

struct EnumState {
    end_ptr: *c_void,
    state: EnumVisitState
}

/// XXX: This should not use a boxed writer!
struct ReprPrinter {
    mut ptr: *c_void,
    writer: @Writer,    // XXX: This should not use a boxed trait.
    enum_stack: DVec<EnumState>
}

/// FIXME (issue #3462): This is horrible.
struct ReprPrinterWrapper {
    printer: @ReprPrinter
}

impl ReprPrinter {
    #[inline(always)]
    fn align(n: uint) {
        unsafe {
            self.ptr = transmute(align(self.ptr as uint, n));
        }
    }

    #[inline(always)]
    fn bump(n: uint) {
        unsafe {
            self.ptr = transmute(self.ptr as uint + n);
        }
    }

    #[inline(always)]
    fn log_simple<T:Repr>() -> bool {
        unsafe {
            self.align(sys::min_align_of::<T>());
            let value_addr: &T = transmute(copy self.ptr);
            value_addr.write_repr(self.writer);
            self.bump(sys::size_of::<T>());
            true
        }
    }
}

impl ReprPrinterWrapper {
    fn visit_estr() -> bool {
        unsafe {
            self.printer.writer.write_char('"');
            let vec_repr_ptr: **VecRepr = transmute(copy self.printer.ptr);
            let vec_repr = *vec_repr_ptr;
            let data_ptr = ptr::to_unsafe_ptr(&(*vec_repr).unboxed.data);
            let slice: &str = transmute((data_ptr, (*vec_repr).unboxed.fill));
            for str::chars_each(slice) |ch| {
                self.printer.writer.write_escaped_char(ch);
            }
            self.printer.writer.write_char('"');
            let ptr_size = sys::size_of::<*c_void>();
            self.printer.ptr = transmute(self.printer.ptr as uint + ptr_size);
            true
        }
    }

    fn visit_self_describing_heap_alloc(mtbl: uint) -> bool {
        unsafe {
            if mtbl != 1 { self.printer.writer.write_str("mut "); }
            let box_ptr: **BoxRepr = transmute(copy self.printer.ptr);
            let box = *box_ptr;
            self.printer.ptr = transmute(&(*box).data);
            intrinsic::visit_tydesc((*box).header.type_desc,
                                    self as @TyVisitor);
            let box_size = sys::size_of::<*BoxRepr>();
            self.printer.ptr = transmute(box_ptr as uint + box_size);
            true
        }
    }

    fn visit_ptr_contents(mtbl: uint, inner: *TyDesc) -> bool {
        unsafe {
            if mtbl != 1 { self.printer.writer.write_str("mut "); }
            let data_ptr: **c_void = transmute(copy self.printer.ptr);
            if *data_ptr == ptr::null() {
                self.printer.writer.write_str("null");
            } else {
                self.printer.ptr = *data_ptr;
                intrinsic::visit_tydesc(inner, self as @TyVisitor);
            }
            let ptr_size = sys::size_of::<*c_void>();
            self.printer.ptr = transmute(data_ptr as uint + ptr_size);
            true
        }
    }

    fn visit_evec(mtbl: uint, inner: *TyDesc) -> bool {
        unsafe {
            self.printer.writer.write_char('[');
            self.printer.align(sys::min_align_of::<*c_void>());
            let vec_repr_ptr: **VecRepr = transmute(copy self.printer.ptr);
            let old_ptr = self.printer.ptr as uint;
            let vec_repr: *VecRepr = *vec_repr_ptr;
            self.printer.ptr = transmute(&(*vec_repr).unboxed.data);
            let end_ptr: *c_void = transmute(self.printer.ptr as uint +
                                             (*vec_repr).unboxed.fill);
            let sys_tydesc: *TypeDesc = transmute(copy inner);
            let alignment = (*sys_tydesc).align;
            let mut first = true;
            loop {
                self.printer.align(alignment);
                if self.printer.ptr >= end_ptr { break; }
                if first {
                    self.printer.writer.write_char(' ');
                    if mtbl != 1 { self.printer.writer.write_str("mut "); }
                } else {
                    self.printer.writer.write_str(", ");
                }
                intrinsic::visit_tydesc(inner, self as @TyVisitor);
                first = false;
            }
            if !first {
                self.printer.writer.write_char(' ');
            } else if mtbl != 1 {
                self.printer.writer.write_str("mut");
            }
            self.printer.writer.write_char(']');
            self.printer.ptr = transmute(old_ptr + sys::size_of::<int>());
            true
        }
    }
}

impl ReprPrinterWrapper : TyVisitor {
    fn visit_bot() -> bool {
        self.printer.bump(1);
        self.printer.writer.write_str("fail");
        true
    }

    fn visit_nil() -> bool { self.printer.log_simple::<()>() }
    fn visit_bool() -> bool { self.printer.log_simple::<bool>() }

    // Numbers

    fn visit_int() -> bool { self.printer.log_simple::<int>() }
    fn visit_i8() -> bool { self.printer.log_simple::<i8>() }
    fn visit_i16() -> bool { self.printer.log_simple::<i16>() }
    fn visit_i32() -> bool { self.printer.log_simple::<i32>() }
    fn visit_i64() -> bool { self.printer.log_simple::<i64>() }

    fn visit_uint() -> bool { self.printer.log_simple::<uint>() }
    fn visit_u8() -> bool { self.printer.log_simple::<u8>() }
    fn visit_u16() -> bool { self.printer.log_simple::<u16>() }
    fn visit_u32() -> bool { self.printer.log_simple::<u32>() }
    fn visit_u64() -> bool { self.printer.log_simple::<u64>() }

    fn visit_float() -> bool { self.printer.log_simple::<float>() }
    fn visit_f32() -> bool { self.printer.log_simple::<f32>() }
    fn visit_f64() -> bool { self.printer.log_simple::<f64>() }

    fn visit_char() -> bool { self.printer.log_simple::<char>() }
    fn visit_str() -> bool { true }

    // Strings

    fn visit_estr_box() -> bool {
        self.printer.writer.write_char('@');
        self.visit_estr()
    }
    fn visit_estr_uniq() -> bool {
        self.printer.writer.write_char('~');
        self.visit_estr()
    }
    fn visit_estr_slice() -> bool {
        unsafe {
            self.printer.writer.write_char('"');
            let slice_ptr: *&str = transmute(copy self.printer.ptr);
            let slice = *slice_ptr;
            for str::chars_each(slice) |ch| {
                self.printer.writer.write_escaped_char(ch);
            }
            self.printer.writer.write_char('"');
            let slice_sz = sys::size_of::<(*char,uint)>();
            self.printer.ptr = transmute(self.printer.ptr as uint + slice_sz);
            true
        }
    }
    fn visit_estr_fixed(_n: uint, _sz: uint, _align: uint) -> bool { true }

    // Pointers

    fn visit_box(mtbl: uint, _inner: *TyDesc) -> bool {
        self.printer.writer.write_char('@');
        self.visit_self_describing_heap_alloc(mtbl)
    }
    fn visit_uniq(mtbl: uint, _inner: *TyDesc) -> bool {
        self.printer.writer.write_char('~');
        self.visit_self_describing_heap_alloc(mtbl)
    }
    fn visit_ptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.printer.writer.write_char('*');
        self.visit_ptr_contents(mtbl, inner)
    }
    fn visit_rptr(mtbl: uint, inner: *TyDesc) -> bool {
        self.printer.writer.write_char('&');
        self.visit_ptr_contents(mtbl, inner)
    }

    // Vectors

    fn visit_vec(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_evec_box(mtbl: uint, inner: *TyDesc) -> bool {
        self.printer.writer.write_char('@');
        self.visit_evec(mtbl, inner)
    }
    fn visit_evec_uniq(mtbl: uint, inner: *TyDesc) -> bool {
        self.printer.writer.write_char('~');
        self.visit_evec(mtbl, inner)
    }
    fn visit_evec_slice(mtbl: uint, inner: *TyDesc) -> bool {
        unsafe {
            self.printer.writer.write_str("&[");
            self.printer.align(sys::min_align_of::<(*c_void,uint)>());
            let slice_ptr: *(*c_void,uint) = transmute(copy self.printer.ptr);
            let (data, fill) = *slice_ptr;
            self.printer.ptr = data;
            let end_ptr: *c_void = transmute(self.printer.ptr as uint + fill);
            let sys_tydesc: *TypeDesc = transmute(copy inner);
            let alignment = (*sys_tydesc).align;
            let mut first = true;
            loop {
                self.printer.align(alignment);
                if self.printer.ptr >= end_ptr { break; }
                if first {
                    self.printer.writer.write_char(' ');
                    if mtbl != 1 { self.printer.writer.write_str("mut "); }
                } else {
                    self.printer.writer.write_str(", ");
                }
                intrinsic::visit_tydesc(inner, self as @TyVisitor);
                first = false;
            }
            if !first {
                self.printer.writer.write_char(' ');
            } else if mtbl != 1 {
                self.printer.writer.write_str("mut");
            }
            self.printer.writer.write_char(']');
            let slice_size = sys::size_of::<(uint, *c_void)>();
            self.printer.ptr = transmute(slice_ptr as uint + slice_size);
            true
        }
    }
    fn visit_evec_fixed(n: uint, sz: uint, align: uint, mtbl: uint,
                        inner: *TyDesc) -> bool {
        unsafe {
            self.printer.writer.write_char('[');
            self.printer.align(align);
            let end_ptr: *c_void = transmute(self.printer.ptr as uint + sz);
            for uint::range(0, n) |i| {
                self.printer.align(align);
                if i == 0 {
                    self.printer.writer.write_char(' ');
                    if mtbl != 1 { self.printer.writer.write_str("mut "); }
                } else {
                    self.printer.writer.write_str(", ");
                }
                intrinsic::visit_tydesc(inner, self as @TyVisitor);
            }
            if n > 0 {
                self.printer.writer.write_char(' ');
            } else if mtbl != 1 {
                self.printer.writer.write_str("mut");
            }
            self.printer.writer.write_char(']');
            self.printer.ptr = end_ptr;
            true
        }
    }

    // Records

    fn visit_enter_rec(_n_fields: uint, _sz: uint, align: uint) -> bool {
        self.printer.writer.write_char('{');
        self.printer.align(align);
        true
    }
    fn visit_rec_field(i: uint, name: &str, mtbl: uint, inner: *TyDesc) ->
                       bool {
        if i != 0 {
            self.printer.writer.write_str(", ");
        } else {
            self.printer.writer.write_char(' ');
        }
        if mtbl != 1 { self.printer.writer.write_str("mut "); }
        self.printer.writer.write_str(name);
        self.printer.writer.write_str(": ");
        intrinsic::visit_tydesc(inner, self as @TyVisitor);
        true
    }
    fn visit_leave_rec(n_fields: uint, _sz: uint, _align: uint) -> bool {
        if n_fields > 0 { self.printer.writer.write_char(' '); }
        self.printer.writer.write_char('}');
        true
    }

    // Structs

    fn visit_enter_class(_n_fields: uint, _sz: uint, align: uint) -> bool {
        self.printer.writer.write_char('{');
        self.printer.align(align);
        true
    }
    fn visit_class_field(i: uint, name: &str, mtbl: uint, inner: *TyDesc) ->
                         bool {
        if i != 0 {
            self.printer.writer.write_str(", ");
        } else {
            self.printer.writer.write_char(' ');
        }
        if mtbl != 1 { self.printer.writer.write_str("mut "); }
        self.printer.writer.write_str(name);
        self.printer.writer.write_str(": ");
        intrinsic::visit_tydesc(inner, self as @TyVisitor);
        true
    }
    fn visit_leave_class(n_fields: uint, _sz: uint, _align: uint) -> bool {
        if n_fields > 0 { self.printer.writer.write_char(' '); }
        self.printer.writer.write_char('}');
        true
    }

    // Tuples

    fn visit_enter_tup(_n_fields: uint, _sz: uint, align: uint) -> bool {
        self.printer.writer.write_char('(');
        self.printer.align(align);
        true
    }
    fn visit_tup_field(i: uint, inner: *TyDesc) -> bool {
        if i != 0 { self.printer.writer.write_str(", "); }
        intrinsic::visit_tydesc(inner, self as @TyVisitor);
        true
    }
    fn visit_leave_tup(_n_fields: uint, _sz: uint, _align: uint) -> bool {
        self.printer.writer.write_char(')');
        true
    }

    // Enums

    fn visit_enter_enum(n_variants: uint, sz: uint, align: uint) -> bool {
        unsafe {
            self.printer.align(align);

            // Write in the location of the end of this enum.
            let end_ptr = transmute(self.printer.ptr as uint + sz);
            let state = if n_variants == 1 { Degenerate } else { PreVariant };
            let new_state = EnumState { end_ptr: end_ptr, state: state };
            self.printer.enum_stack.push(new_state);

            true
        }
    }

    fn visit_enter_enum_variant(_variant: uint,
                                disr_val: int,
                                _n_fields: uint,
                                name: &str) -> bool {
        unsafe {
            let stack = &self.printer.enum_stack;
            let mut enum_state = stack.last();
            match enum_state.state {
                PreVariant => {
                    let disr_ptr = self.printer.ptr as *int;
                    if *disr_ptr == disr_val {
                        enum_state.state = InVariant;
                        self.printer.writer.write_str(name);
                        self.printer.bump(sys::size_of::<int>());
                        stack.set_elt(stack.len() - 1, enum_state);
                    }
                }
                Degenerate => {
                    self.printer.writer.write_str(name);
                }
                InVariant | PostVariant => {}
            }
            true
        }
    }

    fn visit_enum_variant_field(i: uint, inner: *TyDesc) -> bool {
        match self.printer.enum_stack.last().state {
            InVariant | Degenerate => {
                if i == 0 {
                    self.printer.writer.write_char('(');
                } else {
                    self.printer.writer.write_str(", ");
                }

                intrinsic::visit_tydesc(inner, self as @TyVisitor);
            }
            PreVariant | PostVariant => {}
        }
        true
    }

    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                n_fields: uint,
                                _name: &str) -> bool {
        let stack = &self.printer.enum_stack;
        let mut enum_state = stack.last();
        match enum_state.state {
            InVariant => {
                if n_fields >= 1 { self.printer.writer.write_char(')'); }
                enum_state.state = PostVariant;
                stack.set_elt(stack.len() - 1, enum_state);
            }
            Degenerate => {
                if n_fields >= 1 { self.printer.writer.write_char(')'); }
            }
            PreVariant | PostVariant => {}
        }
        true
    }

    fn visit_leave_enum(_n_variants: uint, _sz: uint, _align: uint) -> bool {
        self.printer.ptr = self.printer.enum_stack.pop().end_ptr;
        true
    }

    // Functions

    fn visit_enter_fn(_purity: uint, proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool {
        self.printer.align(sys::min_align_of::<(uint,uint)>());
        match proto {
            2u => self.printer.writer.write_char('~'),
            3u => self.printer.writer.write_char('@'),
            4u => self.printer.writer.write_char('&'),
            _ => {}
        }
        self.printer.writer.write_str("fn");
        true
    }

    fn visit_fn_input(_i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(_retstyle: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }

    // Others

    fn visit_trait() -> bool { self.printer.writer.write_str("@trait"); true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }
    fn visit_opaque_box() -> bool {
        self.printer.writer.write_char('@');
        self.visit_self_describing_heap_alloc(1)
    }
    fn visit_constr(_inner: *TyDesc) -> bool { true }
    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

pub fn write_repr<T>(writer: @Writer, object: &T) {
    unsafe {
        let ptr = ptr::to_unsafe_ptr(object) as *c_void;
        let tydesc = sys::get_type_desc::<T>();
        let tydesc = cast::transmute(move tydesc);

        let repr_printer = @ReprPrinter {
            ptr: ptr,
            writer: writer,
            enum_stack: DVec()
        };

        let wrapper = ReprPrinterWrapper { printer: repr_printer };
        intrinsic::visit_tydesc(tydesc, wrapper as @TyVisitor);
    }
}

