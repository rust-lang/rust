use gccjit::LValue;
use gccjit::{RValue, Type, ToRValue};
use rustc_codegen_ssa::mir::place::PlaceRef;
use rustc_codegen_ssa::traits::{
    BaseTypeMethods,
    ConstMethods,
    DerivedTypeMethods,
    MiscMethods,
    StaticMethods,
};
use rustc_middle::mir::Mutability;
use rustc_middle::ty::layout::{TyAndLayout, LayoutOf};
use rustc_middle::mir::interpret::{ConstAllocation, GlobalAlloc, Scalar};
use rustc_target::abi::{self, HasDataLayout, Pointer, Size};

use crate::consts::const_alloc_to_gcc;
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn const_bytes(&self, bytes: &[u8]) -> RValue<'gcc> {
        bytes_in_context(self, bytes)
    }

    fn global_string(&self, string: &str) -> LValue<'gcc> {
        // TODO(antoyo): handle non-null-terminated strings.
        let string = self.context.new_string_literal(&*string);
        let sym = self.generate_local_symbol_name("str");
        let global = self.declare_private_global(&sym, self.val_ty(string));
        global.global_set_initializer_rvalue(string);
        global
        // TODO(antoyo): set linkage.
    }
}

pub fn bytes_in_context<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, bytes: &[u8]) -> RValue<'gcc> {
    let context = &cx.context;
    let byte_type = context.new_type::<u8>();
    let typ = context.new_array_type(None, byte_type, bytes.len() as i32);
    let elements: Vec<_> =
        bytes.iter()
        .map(|&byte| context.new_rvalue_from_int(byte_type, byte as i32))
        .collect();
    context.new_array_constructor(None, typ, &elements)
}

pub fn type_is_pointer<'gcc>(typ: Type<'gcc>) -> bool {
    typ.get_pointee().is_some()
}

impl<'gcc, 'tcx> ConstMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn const_null(&self, typ: Type<'gcc>) -> RValue<'gcc> {
        if type_is_pointer(typ) {
            self.context.new_null(typ)
        }
        else {
            self.const_int(typ, 0)
        }
    }

    fn const_undef(&self, typ: Type<'gcc>) -> RValue<'gcc> {
        let local = self.current_func.borrow().expect("func")
            .new_local(None, typ, "undefined");
        if typ.is_struct().is_some() {
            // NOTE: hack to workaround a limitation of the rustc API: see comment on
            // CodegenCx.structs_as_pointer
            let pointer = local.get_address(None);
            self.structs_as_pointer.borrow_mut().insert(pointer);
            pointer
        }
        else {
            local.to_rvalue()
        }
    }

    fn const_int(&self, typ: Type<'gcc>, int: i64) -> RValue<'gcc> {
        self.gcc_int(typ, int)
    }

    fn const_uint(&self, typ: Type<'gcc>, int: u64) -> RValue<'gcc> {
        self.gcc_uint(typ, int)
    }

    fn const_uint_big(&self, typ: Type<'gcc>, num: u128) -> RValue<'gcc> {
        self.gcc_uint_big(typ, num)
    }

    fn const_bool(&self, val: bool) -> RValue<'gcc> {
        self.const_uint(self.type_i1(), val as u64)
    }

    fn const_i16(&self, i: i16) -> RValue<'gcc> {
        self.const_int(self.type_i16(), i as i64)
    }

    fn const_i32(&self, i: i32) -> RValue<'gcc> {
        self.const_int(self.type_i32(), i as i64)
    }

    fn const_u32(&self, i: u32) -> RValue<'gcc> {
        self.const_uint(self.type_u32(), i as u64)
    }

    fn const_u64(&self, i: u64) -> RValue<'gcc> {
        self.const_uint(self.type_u64(), i)
    }

    fn const_usize(&self, i: u64) -> RValue<'gcc> {
        let bit_size = self.data_layout().pointer_size.bits();
        if bit_size < 64 {
            // make sure it doesn't overflow
            assert!(i < (1 << bit_size));
        }

        self.const_uint(self.usize_type, i)
    }

    fn const_u8(&self, _i: u8) -> RValue<'gcc> {
        unimplemented!();
    }

    fn const_real(&self, typ: Type<'gcc>, val: f64) -> RValue<'gcc> {
        self.context.new_rvalue_from_double(typ, val)
    }

    fn const_str(&self, s: &str) -> (RValue<'gcc>, RValue<'gcc>) {
        let str_global = *self
            .const_str_cache
            .borrow_mut()
            .raw_entry_mut()
            .from_key(s)
            .or_insert_with(|| (s.to_owned(), self.global_string(s)))
            .1;
        let len = s.len();
        let cs = self.const_ptrcast(str_global.get_address(None),
            self.type_ptr_to(self.layout_of(self.tcx.types.str_).gcc_type(self, true)),
        );
        (cs, self.const_usize(len as u64))
    }

    fn const_struct(&self, values: &[RValue<'gcc>], packed: bool) -> RValue<'gcc> {
        let fields: Vec<_> = values.iter()
            .map(|value| value.get_type())
            .collect();
        // TODO(antoyo): cache the type? It's anonymous, so probably not.
        let typ = self.type_struct(&fields, packed);
        let struct_type = typ.is_struct().expect("struct type");
        self.context.new_struct_constructor(None, struct_type.as_type(), None, values)
    }

    fn const_to_opt_uint(&self, _v: RValue<'gcc>) -> Option<u64> {
        // TODO(antoyo)
        None
    }

    fn const_to_opt_u128(&self, _v: RValue<'gcc>, _sign_ext: bool) -> Option<u128> {
        // TODO(antoyo)
        None
    }

    fn scalar_to_backend(&self, cv: Scalar, layout: abi::Scalar, ty: Type<'gcc>) -> RValue<'gcc> {
        let bitsize = if layout.is_bool() { 1 } else { layout.size(self).bits() };
        match cv {
            Scalar::Int(int) => {
                let data = int.assert_bits(layout.size(self));

                // FIXME(antoyo): there's some issues with using the u128 code that follows, so hard-code
                // the paths for floating-point values.
                if ty == self.float_type {
                    return self.context.new_rvalue_from_double(ty, f32::from_bits(data as u32) as f64);
                }
                else if ty == self.double_type {
                    return self.context.new_rvalue_from_double(ty, f64::from_bits(data as u64));
                }

                let value = self.const_uint_big(self.type_ix(bitsize), data);
                // TODO(bjorn3): assert size is correct
                self.const_bitcast(value, ty)
            }
            Scalar::Ptr(ptr, _size) => {
                let (alloc_id, offset) = ptr.into_parts();
                let base_addr =
                    match self.tcx.global_alloc(alloc_id) {
                        GlobalAlloc::Memory(alloc) => {
                            let init = const_alloc_to_gcc(self, alloc);
                            let alloc = alloc.inner();
                            let value =
                                match alloc.mutability {
                                    Mutability::Mut => self.static_addr_of_mut(init, alloc.align, None),
                                    _ => self.static_addr_of(init, alloc.align, None),
                                };
                            if !self.sess().fewer_names() {
                                // TODO(antoyo): set value name.
                            }
                            value
                        },
                        GlobalAlloc::Function(fn_instance) => {
                            self.get_fn_addr(fn_instance)
                        },
                        GlobalAlloc::VTable(ty, trait_ref) => {
                            let alloc = self.tcx.global_alloc(self.tcx.vtable_allocation((ty, trait_ref))).unwrap_memory();
                            let init = const_alloc_to_gcc(self, alloc);
                            self.static_addr_of(init, alloc.inner().align, None)
                        }
                        GlobalAlloc::Static(def_id) => {
                            assert!(self.tcx.is_static(def_id));
                            self.get_static(def_id).get_address(None)
                        },
                    };
                let ptr_type = base_addr.get_type();
                let base_addr = self.const_bitcast(base_addr, self.usize_type);
                let offset = self.context.new_rvalue_from_long(self.usize_type, offset.bytes() as i64);
                let ptr = self.const_bitcast(base_addr + offset, ptr_type);
                if layout.primitive() != Pointer {
                    self.const_bitcast(ptr.dereference(None).to_rvalue(), ty)
                }
                else {
                    self.const_bitcast(ptr, ty)
                }
            }
        }
    }

    fn const_data_from_alloc(&self, alloc: ConstAllocation<'tcx>) -> Self::Value {
        const_alloc_to_gcc(self, alloc)
    }

    fn from_const_alloc(&self, layout: TyAndLayout<'tcx>, alloc: ConstAllocation<'tcx>, offset: Size) -> PlaceRef<'tcx, RValue<'gcc>> {
        assert_eq!(alloc.inner().align, layout.align.abi);
        let ty = self.type_ptr_to(layout.gcc_type(self, true));
        let value =
            if layout.size == Size::ZERO {
                let value = self.const_usize(alloc.inner().align.bytes());
                self.context.new_cast(None, value, ty)
            }
            else {
                let init = const_alloc_to_gcc(self, alloc);
                let base_addr = self.static_addr_of(init, alloc.inner().align, None);

                let array = self.const_bitcast(base_addr, self.type_i8p());
                let value = self.context.new_array_access(None, array, self.const_usize(offset.bytes())).get_address(None);
                self.const_bitcast(value, ty)
            };
        PlaceRef::new_sized(value, layout)
    }

    fn const_ptrcast(&self, val: RValue<'gcc>, ty: Type<'gcc>) -> RValue<'gcc> {
        self.context.new_cast(None, val, ty)
    }
}

pub trait SignType<'gcc, 'tcx> {
    fn is_signed(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_unsigned(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn to_signed(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn to_unsigned(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
}

impl<'gcc, 'tcx> SignType<'gcc, 'tcx> for Type<'gcc> {
    fn is_signed(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.is_i8(cx) || self.is_i16(cx) || self.is_i32(cx) || self.is_i64(cx) || self.is_i128(cx)
    }

    fn is_unsigned(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.is_u8(cx) || self.is_u16(cx) || self.is_u32(cx) || self.is_u64(cx) || self.is_u128(cx)
    }

    fn to_signed(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        if self.is_u8(cx) {
            cx.i8_type
        }
        else if self.is_u16(cx) {
            cx.i16_type
        }
        else if self.is_u32(cx) {
            cx.i32_type
        }
        else if self.is_u64(cx) {
            cx.i64_type
        }
        else if self.is_u128(cx) {
            cx.i128_type
        }
        else if self.is_uchar(cx) {
            cx.char_type
        }
        else if self.is_ushort(cx) {
            cx.short_type
        }
        else if self.is_uint(cx) {
            cx.int_type
        }
        else if self.is_ulong(cx) {
            cx.long_type
        }
        else if self.is_ulonglong(cx) {
            cx.longlong_type
        }
        else {
            self.clone()
        }
    }

    fn to_unsigned(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        if self.is_i8(cx) {
            cx.u8_type
        }
        else if self.is_i16(cx) {
            cx.u16_type
        }
        else if self.is_i32(cx) {
            cx.u32_type
        }
        else if self.is_i64(cx) {
            cx.u64_type
        }
        else if self.is_i128(cx) {
            cx.u128_type
        }
        else if self.is_char(cx) {
            cx.uchar_type
        }
        else if self.is_short(cx) {
            cx.ushort_type
        }
        else if self.is_int(cx) {
            cx.uint_type
        }
        else if self.is_long(cx) {
            cx.ulong_type
        }
        else if self.is_longlong(cx) {
            cx.ulonglong_type
        }
        else {
            self.clone()
        }
    }
}

pub trait TypeReflection<'gcc, 'tcx>  {
    fn is_uchar(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_ushort(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_uint(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_ulong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_ulonglong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_char(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_short(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_int(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_long(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_longlong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;

    fn is_i8(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_u8(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_i16(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_u16(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_i32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_u32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_i64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_u64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_i128(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_u128(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;

    fn is_f32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;
    fn is_f64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool;

    fn is_vector(&self) -> bool;
}

impl<'gcc, 'tcx> TypeReflection<'gcc, 'tcx> for Type<'gcc> {
    fn is_uchar(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.uchar_type
    }

    fn is_ushort(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.ushort_type
    }

    fn is_uint(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.uint_type
    }

    fn is_ulong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.ulong_type
    }

    fn is_ulonglong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.ulonglong_type
    }

    fn is_char(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.char_type
    }

    fn is_short(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.short_type
    }

    fn is_int(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.int_type
    }

    fn is_long(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.long_type
    }

    fn is_longlong(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.longlong_type
    }

    fn is_i8(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.i8_type
    }

    fn is_u8(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.u8_type
    }

    fn is_i16(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.i16_type
    }

    fn is_u16(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.u16_type
    }

    fn is_i32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.i32_type
    }

    fn is_u32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.u32_type
    }

    fn is_i64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.i64_type
    }

    fn is_u64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.u64_type
    }

    fn is_i128(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.i128_type.unqualified()
    }

    fn is_u128(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.u128_type.unqualified()
    }

    fn is_f32(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.context.new_type::<f32>()
    }

    fn is_f64(&self, cx: &CodegenCx<'gcc, 'tcx>) -> bool {
        self.unqualified() == cx.context.new_type::<f64>()
    }

    fn is_vector(&self) -> bool {
        let mut typ = self.clone();
        loop {
            if typ.dyncast_vector().is_some() {
                return true;
            }

            let old_type = typ;
            typ = typ.unqualified();
            if old_type == typ {
                break;
            }
        }

        false
    }
}
