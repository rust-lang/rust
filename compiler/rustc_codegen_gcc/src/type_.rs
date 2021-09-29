use std::convert::TryInto;

use gccjit::{RValue, Struct, Type};
use rustc_codegen_ssa::traits::{BaseTypeMethods, DerivedTypeMethods};
use rustc_codegen_ssa::common::TypeKind;
use rustc_middle::bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::abi::{AddressSpace, Align, Integer, Size};

use crate::common::TypeReflection;
use crate::context::CodegenCx;
use crate::type_of::LayoutGccExt;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn type_ix(&self, num_bits: u64) -> Type<'gcc> {
        // gcc only supports 1, 2, 4 or 8-byte integers.
        // FIXME(antoyo): this is misleading to use the next power of two as rustc_codegen_ssa
        // sometimes use 96-bit numbers and the following code will give an integer of a different
        // size.
        let bytes = (num_bits / 8).next_power_of_two() as i32;
        match bytes {
            1 => self.i8_type,
            2 => self.i16_type,
            4 => self.i32_type,
            8 => self.i64_type,
            16 => self.i128_type,
            _ => panic!("unexpected num_bits: {}", num_bits),
        }
    }

    pub fn type_void(&self) -> Type<'gcc> {
        self.context.new_type::<()>()
    }

    pub fn type_size_t(&self) -> Type<'gcc> {
        self.context.new_type::<usize>()
    }

    pub fn type_u8(&self) -> Type<'gcc> {
        self.u8_type
    }

    pub fn type_u16(&self) -> Type<'gcc> {
        self.u16_type
    }

    pub fn type_u32(&self) -> Type<'gcc> {
        self.u32_type
    }

    pub fn type_u64(&self) -> Type<'gcc> {
        self.u64_type
    }

    pub fn type_u128(&self) -> Type<'gcc> {
        self.u128_type
    }

    pub fn type_pointee_for_align(&self, align: Align) -> Type<'gcc> {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = Integer::approximate_align(self, align);
        self.type_from_integer(ity)
    }
}

impl<'gcc, 'tcx> BaseTypeMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn type_i1(&self) -> Type<'gcc> {
        self.bool_type
    }

    fn type_i8(&self) -> Type<'gcc> {
        self.i8_type
    }

    fn type_i16(&self) -> Type<'gcc> {
        self.i16_type
    }

    fn type_i32(&self) -> Type<'gcc> {
        self.i32_type
    }

    fn type_i64(&self) -> Type<'gcc> {
        self.i64_type
    }

    fn type_i128(&self) -> Type<'gcc> {
        self.i128_type
    }

    fn type_isize(&self) -> Type<'gcc> {
        self.isize_type
    }

    fn type_f32(&self) -> Type<'gcc> {
        self.context.new_type::<f32>()
    }

    fn type_f64(&self) -> Type<'gcc> {
        self.context.new_type::<f64>()
    }

    fn type_func(&self, params: &[Type<'gcc>], return_type: Type<'gcc>) -> Type<'gcc> {
        self.context.new_function_pointer_type(None, return_type, params, false)
    }

    fn type_struct(&self, fields: &[Type<'gcc>], _packed: bool) -> Type<'gcc> {
        let types = fields.to_vec();
        if let Some(typ) = self.struct_types.borrow().get(fields) {
            return typ.clone();
        }
        let fields: Vec<_> = fields.iter().enumerate()
            .map(|(index, field)| self.context.new_field(None, *field, &format!("field{}_TODO", index)))
            .collect();
        // TODO(antoyo): use packed.
        let typ = self.context.new_struct_type(None, "struct", &fields).as_type();
        self.struct_types.borrow_mut().insert(types, typ);
        typ
    }

    fn type_kind(&self, typ: Type<'gcc>) -> TypeKind {
        if typ.is_integral() {
            TypeKind::Integer
        }
        else if typ.is_vector().is_some() {
            TypeKind::Vector
        }
        else {
            // TODO(antoyo): support other types.
            TypeKind::Void
        }
    }

    fn type_ptr_to(&self, ty: Type<'gcc>) -> Type<'gcc> {
        ty.make_pointer()
    }

    fn type_ptr_to_ext(&self, ty: Type<'gcc>, _address_space: AddressSpace) -> Type<'gcc> {
        // TODO(antoyo): use address_space
        ty.make_pointer()
    }

    fn element_type(&self, ty: Type<'gcc>) -> Type<'gcc> {
        if let Some(typ) = ty.is_array() {
            typ
        }
        else if let Some(vector_type) = ty.is_vector() {
            vector_type.get_element_type()
        }
        else if let Some(typ) = ty.get_pointee() {
            typ
        }
        else {
            unreachable!()
        }
    }

    fn vector_length(&self, _ty: Type<'gcc>) -> usize {
        unimplemented!();
    }

    fn float_width(&self, typ: Type<'gcc>) -> usize {
        let f32 = self.context.new_type::<f32>();
        let f64 = self.context.new_type::<f64>();
        if typ == f32 {
            32
        }
        else if typ == f64 {
            64
        }
        else {
            panic!("Cannot get width of float type {:?}", typ);
        }
        // TODO(antoyo): support other sizes.
    }

    fn int_width(&self, typ: Type<'gcc>) -> u64 {
        if typ.is_i8(self) || typ.is_u8(self) {
            8
        }
        else if typ.is_i16(self) || typ.is_u16(self) {
            16
        }
        else if typ.is_i32(self) || typ.is_u32(self) {
            32
        }
        else if typ.is_i64(self) || typ.is_u64(self) {
            64
        }
        else if typ.is_i128(self) || typ.is_u128(self) {
            128
        }
        else {
            panic!("Cannot get width of int type {:?}", typ);
        }
    }

    fn val_ty(&self, value: RValue<'gcc>) -> Type<'gcc> {
        value.get_type()
    }
}

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    pub fn type_padding_filler(&self, size: Size, align: Align) -> Type<'gcc> {
        let unit = Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    pub fn set_struct_body(&self, typ: Struct<'gcc>, fields: &[Type<'gcc>], _packed: bool) {
        // TODO(antoyo): use packed.
        let fields: Vec<_> = fields.iter().enumerate()
            .map(|(index, field)| self.context.new_field(None, *field, &format!("field_{}", index)))
            .collect();
        typ.set_fields(None, &fields);
    }

    pub fn type_named_struct(&self, name: &str) -> Struct<'gcc> {
        self.context.new_opaque_struct_type(None, name)
    }

    pub fn type_array(&self, ty: Type<'gcc>, mut len: u64) -> Type<'gcc> {
        if let Some(struct_type) = ty.is_struct() {
            if struct_type.get_field_count() == 0 {
                // NOTE: since gccjit only supports i32 for the array size and libcore's tests uses a
                // size of usize::MAX in test_binary_search, we workaround this by setting the size to
                // zero for ZSTs.
                // FIXME(antoyo): fix gccjit API.
                len = 0;
            }
        }

        // NOTE: see note above. Some other test uses usize::MAX.
        if len == u64::MAX {
            len = 0;
        }

        let len: i32 = len.try_into().expect("array len");

        self.context.new_array_type(None, ty, len)
    }
}

pub fn struct_fields<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, layout: TyAndLayout<'tcx>) -> (Vec<Type<'gcc>>, bool) {
    let field_count = layout.fields.count();

    let mut packed = false;
    let mut offset = Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let target_offset = layout.fields.offset(i as usize);
        let field = layout.field(cx, i);
        let effective_field_align =
            layout.align.abi.min(field.align.abi).restrict_for_offset(target_offset);
        packed |= effective_field_align < field.align.abi;

        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        let padding_align = prev_effective_align.min(effective_field_align);
        assert_eq!(offset.align_to(padding_align) + padding, target_offset);
        result.push(cx.type_padding_filler(padding, padding_align));

        result.push(field.gcc_type(cx, !field.ty.is_any_ptr())); // FIXME(antoyo): might need to check if the type is inside another, like Box<Type>.
        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }
    if !layout.is_unsized() && field_count > 0 {
        if offset > layout.size {
            bug!("layout: {:#?} stride: {:?} offset: {:?}", layout, layout.size, offset);
        }
        let padding = layout.size - offset;
        let padding_align = prev_effective_align;
        assert_eq!(offset.align_to(padding_align) + padding, layout.size);
        result.push(cx.type_padding_filler(padding, padding_align));
        assert_eq!(result.len(), 1 + field_count * 2);
    }

    (result, packed)
}
