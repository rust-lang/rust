//! Conversion of internal Rust compiler `rustc_target::abi` and `rustc_abi` items to stable ones.

#![allow(rustc::usage_of_qualified_ty)]

use crate::rustc_smir::{Stable, Tables};
use rustc_middle::ty;
use rustc_target::abi::call::Conv;
use stable_mir::abi::{
    ArgAbi, CallConvention, FieldsShape, FnAbi, Layout, LayoutShape, PassMode, TagEncoding,
    TyAndLayout, ValueAbi, VariantsShape,
};
use stable_mir::ty::{Align, IndexedVal, Size, VariantIdx};
use stable_mir::{opaque, Opaque};

impl<'tcx> Stable<'tcx> for rustc_target::abi::VariantIdx {
    type T = VariantIdx;
    fn stable(&self, _: &mut Tables<'tcx>) -> Self::T {
        VariantIdx::to_val(self.as_usize())
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Endian {
    type T = stable_mir::target::Endian;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            rustc_abi::Endian::Little => stable_mir::target::Endian::Little,
            rustc_abi::Endian::Big => stable_mir::target::Endian::Big,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::TyAndLayout<'tcx, ty::Ty<'tcx>> {
    type T = TyAndLayout;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        TyAndLayout { ty: self.ty.stable(tables), layout: self.layout.stable(tables) }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::Layout<'tcx> {
    type T = Layout;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        tables.layout_id(*self)
    }
}

impl<'tcx> Stable<'tcx>
    for rustc_abi::LayoutS<rustc_target::abi::FieldIdx, rustc_target::abi::VariantIdx>
{
    type T = LayoutShape;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        LayoutShape {
            fields: self.fields.stable(tables),
            variants: self.variants.stable(tables),
            abi: self.abi.stable(tables),
            abi_align: self.align.abi.stable(tables),
            size: self.size.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::call::FnAbi<'tcx, ty::Ty<'tcx>> {
    type T = FnAbi;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        assert!(self.args.len() >= self.fixed_count as usize);
        assert!(!self.c_variadic || matches!(self.conv, Conv::C));
        FnAbi {
            args: self.args.as_ref().stable(tables),
            ret: self.ret.stable(tables),
            fixed_count: self.fixed_count,
            conv: self.conv.stable(tables),
            c_variadic: self.c_variadic,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::call::ArgAbi<'tcx, ty::Ty<'tcx>> {
    type T = ArgAbi;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        ArgAbi {
            ty: self.layout.ty.stable(tables),
            layout: self.layout.layout.stable(tables),
            mode: self.mode.stable(tables),
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::call::Conv {
    type T = CallConvention;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            Conv::C => CallConvention::C,
            Conv::Rust => CallConvention::Rust,
            Conv::Cold => CallConvention::Cold,
            Conv::PreserveMost => CallConvention::PreserveMost,
            Conv::PreserveAll => CallConvention::PreserveAll,
            Conv::ArmAapcs => CallConvention::ArmAapcs,
            Conv::CCmseNonSecureCall => CallConvention::CCmseNonSecureCall,
            Conv::Msp430Intr => CallConvention::Msp430Intr,
            Conv::PtxKernel => CallConvention::PtxKernel,
            Conv::X86Fastcall => CallConvention::X86Fastcall,
            Conv::X86Intr => CallConvention::X86Intr,
            Conv::X86Stdcall => CallConvention::X86Stdcall,
            Conv::X86ThisCall => CallConvention::X86ThisCall,
            Conv::X86VectorCall => CallConvention::X86VectorCall,
            Conv::X86_64SysV => CallConvention::X86_64SysV,
            Conv::X86_64Win64 => CallConvention::X86_64Win64,
            Conv::AmdGpuKernel => CallConvention::AmdGpuKernel,
            Conv::AvrInterrupt => CallConvention::AvrInterrupt,
            Conv::AvrNonBlockingInterrupt => CallConvention::AvrNonBlockingInterrupt,
            Conv::RiscvInterrupt { .. } => CallConvention::RiscvInterrupt,
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_target::abi::call::PassMode {
    type T = PassMode;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            rustc_target::abi::call::PassMode::Ignore => PassMode::Ignore,
            rustc_target::abi::call::PassMode::Direct(attr) => PassMode::Direct(opaque(attr)),
            rustc_target::abi::call::PassMode::Pair(first, second) => {
                PassMode::Pair(opaque(first), opaque(second))
            }
            rustc_target::abi::call::PassMode::Cast { pad_i32, cast } => {
                PassMode::Cast { pad_i32: *pad_i32, cast: opaque(cast) }
            }
            rustc_target::abi::call::PassMode::Indirect { attrs, meta_attrs, on_stack } => {
                PassMode::Indirect {
                    attrs: opaque(attrs),
                    meta_attrs: opaque(meta_attrs),
                    on_stack: *on_stack,
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::FieldsShape<rustc_target::abi::FieldIdx> {
    type T = FieldsShape;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

impl<'tcx> Stable<'tcx>
    for rustc_abi::Variants<rustc_target::abi::FieldIdx, rustc_target::abi::VariantIdx>
{
    type T = VariantsShape;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match self {
            rustc_abi::Variants::Single { index } => {
                VariantsShape::Single { index: index.stable(tables) }
            }
            rustc_abi::Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
                VariantsShape::Multiple {
                    tag: tag.stable(tables),
                    tag_encoding: tag_encoding.stable(tables),
                    tag_field: *tag_field,
                    variants: variants.iter().as_slice().stable(tables),
                }
            }
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::TagEncoding<rustc_target::abi::VariantIdx> {
    type T = TagEncoding;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
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

impl<'tcx> Stable<'tcx> for rustc_abi::Abi {
    type T = ValueAbi;

    fn stable(&self, tables: &mut Tables<'tcx>) -> Self::T {
        match *self {
            rustc_abi::Abi::Uninhabited => ValueAbi::Uninhabited,
            rustc_abi::Abi::Scalar(scalar) => ValueAbi::Scalar(scalar.stable(tables)),
            rustc_abi::Abi::ScalarPair(first, second) => {
                ValueAbi::ScalarPair(first.stable(tables), second.stable(tables))
            }
            rustc_abi::Abi::Vector { element, count } => {
                ValueAbi::Vector { element: element.stable(tables), count }
            }
            rustc_abi::Abi::Aggregate { sized } => ValueAbi::Aggregate { sized },
        }
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Size {
    type T = Size;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        self.bytes_usize()
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Align {
    type T = Align;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        self.bytes()
    }
}

impl<'tcx> Stable<'tcx> for rustc_abi::Scalar {
    type T = Opaque;

    fn stable(&self, _tables: &mut Tables<'tcx>) -> Self::T {
        opaque(self)
    }
}
