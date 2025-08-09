//! Conversion of internal Rust compiler `rustc_target` and `rustc_abi` items to stable ones.

#![allow(rustc::usage_of_qualified_ty)]

use rustc_abi::{ArmCall, CanonAbi, InterruptKind, X86Call};
use rustc_middle::ty;
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_target::callconv;

use crate::abi::{
    AddressSpace, ArgAbi, CallConvention, FieldsShape, FloatLength, FnAbi, IntegerLength,
    IntegerType, Layout, LayoutShape, PassMode, Primitive, ReprFlags, ReprOptions, Scalar,
    TagEncoding, TyAndLayout, ValueAbi, VariantsShape, WrappingRange,
};
use crate::compiler_interface::BridgeTys;
use crate::target::MachineSize as Size;
use crate::ty::{Align, VariantIdx};
use crate::unstable::Stable;
use crate::{IndexedVal, opaque};

impl<'tcx> Stable<'tcx> for rustc_abi::VariantIdx {
    type T = VariantIdx;
    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        VariantIdx::to_val(self.as_usize())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Endian {
    type T = crate::target::Endian;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            rustc_abi::Endian::Little => crate::target::Endian::Little,
            rustc_abi::Endian::Big => crate::target::Endian::Big,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::TyAndLayout<'tcx, ty::Ty<'tcx>> {
    type T = TyAndLayout;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        TyAndLayout { ty: self.ty.stable(tables, cx), layout: self.layout.stable(tables, cx) }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Layout<'tcx> {
    type T = Layout;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        tables.layout_id(cx.lift(*self).unwrap())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::LayoutData<rustc_abi::FieldIdx, rustc_abi::VariantIdx> {
    type T = LayoutShape;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        LayoutShape {
            fields: self.fields.stable(tables, cx),
            variants: self.variants.stable(tables, cx),
            abi: self.backend_repr.stable(tables, cx),
            abi_align: self.align.abi.stable(tables, cx),
            size: self.size.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::FnAbi<'tcx, ty::Ty<'tcx>> {
    type T = FnAbi;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        assert!(self.args.len() >= self.fixed_count as usize);
        assert!(!self.c_variadic || matches!(self.conv, CanonAbi::C));
        FnAbi {
            args: self.args.as_ref().stable(tables, cx),
            ret: self.ret.stable(tables, cx),
            fixed_count: self.fixed_count,
            conv: self.conv.stable(tables, cx),
            c_variadic: self.c_variadic,
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::ArgAbi<'tcx, ty::Ty<'tcx>> {
    type T = ArgAbi;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        ArgAbi {
            ty: self.layout.ty.stable(tables, cx),
            layout: self.layout.layout.stable(tables, cx),
            mode: self.mode.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for CanonAbi {
    type T = CallConvention;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            CanonAbi::C => CallConvention::C,
            CanonAbi::Rust => CallConvention::Rust,
            CanonAbi::RustCold => CallConvention::Cold,
            CanonAbi::Custom => CallConvention::Custom,
            CanonAbi::Arm(arm_call) => match arm_call {
                ArmCall::Aapcs => CallConvention::ArmAapcs,
                ArmCall::CCmseNonSecureCall => CallConvention::CCmseNonSecureCall,
                ArmCall::CCmseNonSecureEntry => CallConvention::CCmseNonSecureEntry,
            },
            CanonAbi::GpuKernel => CallConvention::GpuKernel,
            CanonAbi::Interrupt(interrupt_kind) => match interrupt_kind {
                InterruptKind::Avr => CallConvention::AvrInterrupt,
                InterruptKind::AvrNonBlocking => CallConvention::AvrNonBlockingInterrupt,
                InterruptKind::Msp430 => CallConvention::Msp430Intr,
                InterruptKind::RiscvMachine | InterruptKind::RiscvSupervisor => {
                    CallConvention::RiscvInterrupt
                }
                InterruptKind::X86 => CallConvention::X86Intr,
            },
            CanonAbi::X86(x86_call) => match x86_call {
                X86Call::Fastcall => CallConvention::X86Fastcall,
                X86Call::Stdcall => CallConvention::X86Stdcall,
                X86Call::SysV64 => CallConvention::X86_64SysV,
                X86Call::Thiscall => CallConvention::X86ThisCall,
                X86Call::Vectorcall => CallConvention::X86VectorCall,
                X86Call::Win64 => CallConvention::X86_64Win64,
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::PassMode {
    type T = PassMode;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            callconv::PassMode::Ignore => PassMode::Ignore,
            callconv::PassMode::Direct(attr) => PassMode::Direct(opaque(attr)),
            callconv::PassMode::Pair(first, second) => {
                PassMode::Pair(opaque(first), opaque(second))
            }
            callconv::PassMode::Cast { pad_i32, cast } => {
                PassMode::Cast { pad_i32: *pad_i32, cast: opaque(cast) }
            }
            callconv::PassMode::Indirect { attrs, meta_attrs, on_stack } => PassMode::Indirect {
                attrs: opaque(attrs),
                meta_attrs: opaque(meta_attrs),
                on_stack: *on_stack,
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::FieldsShape<rustc_abi::FieldIdx> {
    type T = FieldsShape;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::FieldsShape::Primitive => FieldsShape::Primitive,
            rustc_abi::FieldsShape::Union(count) => FieldsShape::Union(*count),
            rustc_abi::FieldsShape::Array { stride, count } => {
                FieldsShape::Array { stride: stride.stable(tables, cx), count: *count }
            }
            rustc_abi::FieldsShape::Arbitrary { offsets, .. } => {
                FieldsShape::Arbitrary { offsets: offsets.iter().as_slice().stable(tables, cx) }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Variants<rustc_abi::FieldIdx, rustc_abi::VariantIdx> {
    type T = VariantsShape;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::Variants::Single { index } => {
                VariantsShape::Single { index: index.stable(tables, cx) }
            }
            rustc_abi::Variants::Empty => VariantsShape::Empty,
            rustc_abi::Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                VariantsShape::Multiple {
                    tag: tag.stable(tables, cx),
                    tag_encoding: tag_encoding.stable(tables, cx),
                    tag_field: tag_field.stable(tables, cx),
                    variants: variants.iter().as_slice().stable(tables, cx),
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::TagEncoding<rustc_abi::VariantIdx> {
    type T = TagEncoding;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::TagEncoding::Direct => TagEncoding::Direct,
            rustc_abi::TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                TagEncoding::Niche {
                    untagged_variant: untagged_variant.stable(tables, cx),
                    niche_variants: niche_variants.stable(tables, cx),
                    niche_start: *niche_start,
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::BackendRepr {
    type T = ValueAbi;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match *self {
            rustc_abi::BackendRepr::Scalar(scalar) => ValueAbi::Scalar(scalar.stable(tables, cx)),
            rustc_abi::BackendRepr::ScalarPair(first, second) => {
                ValueAbi::ScalarPair(first.stable(tables, cx), second.stable(tables, cx))
            }
            rustc_abi::BackendRepr::SimdVector { element, count } => {
                ValueAbi::Vector { element: element.stable(tables, cx), count }
            }
            rustc_abi::BackendRepr::Memory { sized } => ValueAbi::Aggregate { sized },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Size {
    type T = Size;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        Size::from_bits(self.bits_usize())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Align {
    type T = Align;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        self.bytes()
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Scalar {
    type T = Scalar;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::Scalar::Initialized { value, valid_range } => Scalar::Initialized {
                value: value.stable(tables, cx),
                valid_range: valid_range.stable(tables, cx),
            },
            rustc_abi::Scalar::Union { value } => Scalar::Union { value: value.stable(tables, cx) },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Primitive {
    type T = Primitive;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::Primitive::Int(length, signed) => {
                Primitive::Int { length: length.stable(tables, cx), signed: *signed }
            }
            rustc_abi::Primitive::Float(length) => {
                Primitive::Float { length: length.stable(tables, cx) }
            }
            rustc_abi::Primitive::Pointer(space) => Primitive::Pointer(space.stable(tables, cx)),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::AddressSpace {
    type T = AddressSpace;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        AddressSpace(self.0)
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Integer {
    type T = IntegerLength;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            rustc_abi::Integer::I8 => IntegerLength::I8,
            rustc_abi::Integer::I16 => IntegerLength::I16,
            rustc_abi::Integer::I32 => IntegerLength::I32,
            rustc_abi::Integer::I64 => IntegerLength::I64,
            rustc_abi::Integer::I128 => IntegerLength::I128,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Float {
    type T = FloatLength;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        match self {
            rustc_abi::Float::F16 => FloatLength::F16,
            rustc_abi::Float::F32 => FloatLength::F32,
            rustc_abi::Float::F64 => FloatLength::F64,
            rustc_abi::Float::F128 => FloatLength::F128,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::WrappingRange {
    type T = WrappingRange;

    fn stable(&self, _: &mut Tables<'_, BridgeTys>, _: &CompilerCtxt<'_, BridgeTys>) -> Self::T {
        WrappingRange { start: self.start, end: self.end }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::ReprFlags {
    type T = ReprFlags;

    fn stable<'cx>(
        &self,
        _tables: &mut Tables<'cx, BridgeTys>,
        _cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        ReprFlags {
            is_simd: self.intersects(Self::IS_SIMD),
            is_c: self.intersects(Self::IS_C),
            is_transparent: self.intersects(Self::IS_TRANSPARENT),
            is_linear: self.intersects(Self::IS_LINEAR),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::IntegerType {
    type T = IntegerType;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            rustc_abi::IntegerType::Pointer(signed) => IntegerType::Pointer { is_signed: *signed },
            rustc_abi::IntegerType::Fixed(integer, signed) => {
                IntegerType::Fixed { length: integer.stable(tables, cx), is_signed: *signed }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::ReprOptions {
    type T = ReprOptions;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        ReprOptions {
            int: self.int.map(|int| int.stable(tables, cx)),
            align: self.align.map(|align| align.stable(tables, cx)),
            pack: self.pack.map(|pack| pack.stable(tables, cx)),
            flags: self.flags.stable(tables, cx),
        }
    }
}
