//! Conversion of internal Rust compiler `rustc_target` and `rustc_abi` items to stable ones.

#![allow(rustc::usage_of_qualified_ty)]

use rustc_abi::{ArmCall, CanonAbi, InterruptKind, X86Call};
use rustc_middle::ty;
use rustc_target::callconv;
use stable_mir::abi::{
    AddressSpace, ArgAbi, CallConvention, FieldsShape, FloatLength, FnAbi, IntegerLength, Layout,
    LayoutShape, PassMode, Primitive, Scalar, TagEncoding, TyAndLayout, ValueAbi, VariantsShape,
    WrappingRange,
};
use stable_mir::opaque;
use stable_mir::target::MachineSize as Size;
use stable_mir::ty::{Align, IndexedVal, VariantIdx};

use crate::rustc_smir::{Stable, Tables};
use crate::stable_mir;

impl<'tcx> Stable<'tcx> for rustc_abi::VariantIdx {
    type T = VariantIdx;
    fn stable(&self, _: &mut Tables<'_>) -> Self::T {
        VariantIdx::to_val(self.as_usize())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Endian {
    type T = stable_mir::target::Endian;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::Endian::Little => stable_mir::target::Endian::Little,
            rustc_abi::Endian::Big => stable_mir::target::Endian::Big,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::TyAndLayout<'tcx, ty::Ty<'tcx>> {
    type T = TyAndLayout;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        TyAndLayout { ty: self.ty.stable(tables), layout: self.layout.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Layout<'tcx> {
    type T = Layout;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        tables.layout_id(tables.tcx.lift(*self).unwrap())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::LayoutData<rustc_abi::FieldIdx, rustc_abi::VariantIdx> {
    type T = LayoutShape;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        LayoutShape {
            fields: self.fields.stable(tables),
            variants: self.variants.stable(tables),
            abi: self.backend_repr.stable(tables),
            abi_align: self.align.abi.stable(tables),
            size: self.size.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::FnAbi<'tcx, ty::Ty<'tcx>> {
    type T = FnAbi;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        assert!(self.args.len() >= self.fixed_count as usize);
        assert!(!self.c_variadic || matches!(self.conv, CanonAbi::C));
        FnAbi {
            args: self.args.as_ref().stable(tables),
            ret: self.ret.stable(tables),
            fixed_count: self.fixed_count,
            conv: self.conv.stable(tables),
            c_variadic: self.c_variadic,
        }
    }
}

impl<'tcx> Stable<'tcx> for callconv::ArgAbi<'tcx, ty::Ty<'tcx>> {
    type T = ArgAbi;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        ArgAbi {
            ty: self.layout.ty.stable(tables),
            layout: self.layout.layout.stable(tables),
            mode: self.mode.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for CanonAbi {
    type T = CallConvention;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        match self {
            CanonAbi::C => CallConvention::C,
            CanonAbi::Rust => CallConvention::Rust,
            CanonAbi::RustCold => CallConvention::Cold,
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

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
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

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::FieldsShape::Primitive => FieldsShape::Primitive,
            rustc_abi::FieldsShape::Union(count) => FieldsShape::Union(*count),
            rustc_abi::FieldsShape::Array { stride, count } => {
                FieldsShape::Array { stride: stride.stable(tables), count: *count }
            }
            rustc_abi::FieldsShape::Arbitrary { offsets, .. } => {
                FieldsShape::Arbitrary { offsets: offsets.iter().as_slice().stable(tables) }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Variants<rustc_abi::FieldIdx, rustc_abi::VariantIdx> {
    type T = VariantsShape;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::Variants::Single { index } => {
                VariantsShape::Single { index: index.stable(tables) }
            }
            rustc_abi::Variants::Empty => VariantsShape::Empty,
            rustc_abi::Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                VariantsShape::Multiple {
                    tag: tag.stable(tables),
                    tag_encoding: tag_encoding.stable(tables),
                    tag_field: tag_field.stable(tables),
                    variants: variants.iter().as_slice().stable(tables),
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::TagEncoding<rustc_abi::VariantIdx> {
    type T = TagEncoding;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::TagEncoding::Direct => TagEncoding::Direct,
            rustc_abi::TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                TagEncoding::Niche {
                    untagged_variant: untagged_variant.stable(tables),
                    niche_variants: niche_variants.stable(tables),
                    niche_start: *niche_start,
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::BackendRepr {
    type T = ValueAbi;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match *self {
            rustc_abi::BackendRepr::Scalar(scalar) => ValueAbi::Scalar(scalar.stable(tables)),
            rustc_abi::BackendRepr::ScalarPair(first, second) => {
                ValueAbi::ScalarPair(first.stable(tables), second.stable(tables))
            }
            rustc_abi::BackendRepr::SimdVector { element, count } => {
                ValueAbi::Vector { element: element.stable(tables), count }
            }
            rustc_abi::BackendRepr::Memory { sized } => ValueAbi::Aggregate { sized },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Size {
    type T = Size;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        Size::from_bits(self.bits_usize())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Align {
    type T = Align;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        self.bytes()
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Scalar {
    type T = Scalar;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::Scalar::Initialized { value, valid_range } => Scalar::Initialized {
                value: value.stable(tables),
                valid_range: valid_range.stable(tables),
            },
            rustc_abi::Scalar::Union { value } => Scalar::Union { value: value.stable(tables) },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Primitive {
    type T = Primitive;

    fn stable(&self, tables: &mut Tables<'_>) -> Self::T {
        match self {
            rustc_abi::Primitive::Int(length, signed) => {
                Primitive::Int { length: length.stable(tables), signed: *signed }
            }
            rustc_abi::Primitive::Float(length) => {
                Primitive::Float { length: length.stable(tables) }
            }
            rustc_abi::Primitive::Pointer(space) => Primitive::Pointer(space.stable(tables)),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::AddressSpace {
    type T = AddressSpace;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        AddressSpace(self.0)
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Integer {
    type T = IntegerLength;

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
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

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
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

    fn stable(&self, _tables: &mut Tables<'_>) -> Self::T {
        WrappingRange { start: self.start, end: self.end }
    }
}
