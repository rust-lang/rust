//! Conversion of internal Rust compiler `rustc_target` and `rustc_abi` items to stable ones.

#![allow(rustc::usage_of_qualified_ty)]

use rustc_abi::{ArmCall, CanonAbi, InterruptKind, X86Call};
use rustc_middle::ty;
use rustc_public_bridge::Tables;
use rustc_public_bridge::context::CompilerCtxt;
use rustc_target::callconv;

use crate::IndexedVal;
use crate::abi::{
    AddressSpace, ArgAbi, ArgAttributes, ArgExtension, CallConvention, CastTarget, FieldsShape,
    FloatLength, FnAbi, IntegerLength, IntegerType, Layout, LayoutShape, NumScalableVectors,
    PassMode, Primitive, Reg, RegKind, ReprFlags, ReprOptions, Scalar, TagEncoding, TyAndLayout,
    Uniform, ValueRepr, VariantFields, VariantsShape, WrappingRange,
};
use crate::compiler_interface::BridgeTys;
use crate::target::MachineSize as Size;
use crate::ty::{Align, VariantIdx};
use crate::unstable::Stable;

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
        tables.layout_id(cx.lift(*self))
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
            value_repr: self.backend_repr.stable(tables, cx),
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
            CanonAbi::RustPreserveNone => CallConvention::PreserveNone,
            CanonAbi::RustTail => CallConvention::Tail,
            CanonAbi::Custom => CallConvention::Custom,
            CanonAbi::Swift => CallConvention::Swift,
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

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match self {
            callconv::PassMode::Ignore => PassMode::Ignore,
            callconv::PassMode::Direct(attr) => PassMode::Direct(attr.stable(tables, cx)),
            callconv::PassMode::Pair(first, second) => {
                PassMode::Pair(first.stable(tables, cx), second.stable(tables, cx))
            }
            callconv::PassMode::Cast { pad_i32, cast } => {
                PassMode::Cast { pad_i32: *pad_i32, cast: cast.stable(tables, cx) }
            }
            callconv::PassMode::Indirect { attrs, meta_attrs, on_stack } => PassMode::Indirect {
                attrs: attrs.stable(tables, cx),
                meta_attrs: meta_attrs.map(|a| a.stable(tables, cx)),
                on_stack: *on_stack,
            },
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::CastTarget {
    type T = CastTarget;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        CastTarget {
            prefix: self.prefix.iter().map(|reg| reg.stable(tables, cx)).collect(),
            rest_offset: self.rest_offset.map(|offset| Size::from_bits(offset.bits_usize())),
            rest: self.rest.stable(tables, cx),
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::Uniform {
    type T = Uniform;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        Uniform {
            unit: self.unit.stable(tables, cx),
            total: Size::from_bits(self.total.bits_usize()),
            is_consecutive: self.is_consecutive,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Reg {
    type T = Reg;

    fn stable<'cx>(
        &self,
        _: &mut Tables<'cx, BridgeTys>,
        _: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        Reg {
            kind: match self.kind {
                rustc_abi::RegKind::Integer => RegKind::Integer,
                rustc_abi::RegKind::Float => RegKind::Float,
                rustc_abi::RegKind::Vector { .. } => RegKind::Vector,
            },
            size: Size::from_bits(self.size.bits_usize()),
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::ArgAttributes {
    type T = ArgAttributes;

    fn stable<'cx>(
        &self,
        _: &mut Tables<'cx, BridgeTys>,
        _: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        ArgAttributes {
            arg_ext: match self.arg_ext {
                callconv::ArgExtension::None => ArgExtension::None,
                callconv::ArgExtension::Zext => ArgExtension::Zext,
                callconv::ArgExtension::Sext => ArgExtension::Sext,
            },
            pointee_size: Size::from_bits(self.pointee_size.bits_usize()),
            pointee_align: self.pointee_align.map(|a| a.bytes()),
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
                    variants: variants
                        .iter()
                        .map(|v| VariantFields {
                            offsets: v.field_offsets.iter().as_slice().stable(tables, cx),
                        })
                        .collect(),
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

impl<'tcx> Stable<'tcx> for rustc_abi::NumScalableVectors {
    type T = NumScalableVectors;

    fn stable<'cx>(
        &self,
        _tables: &mut Tables<'cx, BridgeTys>,
        _cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        NumScalableVectors(self.0)
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::BackendRepr {
    type T = ValueRepr;

    fn stable<'cx>(
        &self,
        tables: &mut Tables<'cx, BridgeTys>,
        cx: &CompilerCtxt<'cx, BridgeTys>,
    ) -> Self::T {
        match *self {
            rustc_abi::BackendRepr::Scalar(scalar) => ValueRepr::Scalar(scalar.stable(tables, cx)),
            rustc_abi::BackendRepr::ScalarPair { a: first, b: second, b_offset: second_offset } => {
                ValueRepr::ScalarPair {
                    a: first.stable(tables, cx),
                    b: second.stable(tables, cx),
                    b_offset: second_offset.stable(tables, cx),
                }
            }
            rustc_abi::BackendRepr::SimdVector { element, count } => {
                ValueRepr::Vector { element: element.stable(tables, cx), count }
            }
            rustc_abi::BackendRepr::SimdScalableVector { element, count, number_of_vectors } => {
                ValueRepr::ScalableVector {
                    element: element.stable(tables, cx),
                    count,
                    number_of_vectors: number_of_vectors.stable(tables, cx),
                }
            }
            rustc_abi::BackendRepr::Memory { sized } => ValueRepr::Aggregate { sized },
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
