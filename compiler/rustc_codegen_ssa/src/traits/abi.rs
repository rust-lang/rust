use rustc_middle::bug;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_target::abi::call::CastTarget;

use super::consts::ConstMethods;
use super::type_::BaseTypeMethods;
use super::{BackendTypes, BuilderMethods, LayoutTypeMethods};
use crate::mir::operand::{OperandRef, OperandValue};
use crate::mir::place::PlaceRef;

pub trait AbiBuilderMethods<'tcx>: BackendTypes {
    fn get_param(&mut self, index: usize) -> Self::Value;
}

/// The ABI mandates that the value is passed as a different struct representation.
pub trait CastTargetAbiExt<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    /// Spill and reload it from the stack to convert from the Rust representation to the ABI representation.
    fn cast_rust_abi_to_other(&self, bx: &mut Bx, op: OperandRef<'tcx, Bx::Value>) -> Bx::Value;
    /// Spill and reload it from the stack to convert from the ABI representation to the Rust representation.
    fn cast_other_abi_to_rust(
        &self,
        bx: &mut Bx,
        src: Bx::Value,
        dst: Bx::Value,
        layout: TyAndLayout<'tcx>,
    );
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> CastTargetAbiExt<'a, 'tcx, Bx> for CastTarget {
    fn cast_rust_abi_to_other(&self, bx: &mut Bx, op: OperandRef<'tcx, Bx::Value>) -> Bx::Value {
        let scratch_size = self.unaligned_size(bx);
        let (has_scratch, src, align) = match op.val {
            // If the source already has enough space, we can cast from it directly.
            OperandValue::Ref(place_val) if op.layout.size >= scratch_size => {
                (false, place_val.llval, place_val.align)
            }
            OperandValue::Immediate(_) | OperandValue::Pair(..) | OperandValue::Ref(_) => {
                // When `op.layout.size` is larger than `scratch_size`, the extra space is just padding.
                let scratch = PlaceRef::alloca_size(bx, scratch_size, op.layout);
                let llscratch = scratch.val.llval;
                bx.lifetime_start(llscratch, scratch_size);
                op.val.store(bx, scratch);
                (true, llscratch, scratch.val.align)
            }
            OperandValue::ZeroSized => bug!("ZST value shouldn't be in PassMode::Cast"),
        };
        let cast_ty = bx.cast_backend_type(self);
        let ret = match bx.type_kind(cast_ty) {
            crate::common::TypeKind::Struct | crate::common::TypeKind::Array => {
                let mut index = 0;
                let mut offset = 0;
                let mut target = bx.const_poison(cast_ty);
                for reg in self.prefix.iter().filter_map(|&x| x) {
                    let ptr = if offset == 0 {
                        src
                    } else {
                        bx.inbounds_ptradd(src, bx.const_usize(offset))
                    };
                    let load = bx.load(bx.reg_backend_type(&reg), ptr, align);
                    target = bx.insert_value(target, load, index);
                    index += 1;
                    offset += reg.size.bytes();
                }
                let (rest_count, rem_bytes) = if self.rest.unit.size.bytes() == 0 {
                    (0, 0)
                } else {
                    (
                        self.rest.total.bytes() / self.rest.unit.size.bytes(),
                        self.rest.total.bytes() % self.rest.unit.size.bytes(),
                    )
                };
                for _ in 0..rest_count {
                    let ptr = if offset == 0 {
                        src
                    } else {
                        bx.inbounds_ptradd(src, bx.const_usize(offset))
                    };
                    let load = bx.load(bx.reg_backend_type(&self.rest.unit), ptr, align);
                    target = bx.insert_value(target, load, index);
                    index += 1;
                    offset += self.rest.unit.size.bytes();
                }
                if rem_bytes != 0 {
                    let ptr = bx.inbounds_ptradd(src, bx.const_usize(offset));
                    let load = bx.load(bx.reg_backend_type(&self.rest.unit), ptr, align);
                    target = bx.insert_value(target, load, index);
                }
                target
            }
            ty_kind if bx.type_kind(bx.reg_backend_type(&self.rest.unit)) == ty_kind => {
                bx.load(cast_ty, src, align)
            }
            ty_kind => bug!("cannot cast {ty_kind:?} to the ABI representation in CastTarget"),
        };
        if has_scratch {
            bx.lifetime_end(src, scratch_size);
        }
        ret
    }

    fn cast_other_abi_to_rust(
        &self,
        bx: &mut Bx,
        src: Bx::Value,
        dst: Bx::Value,
        layout: TyAndLayout<'tcx>,
    ) {
        let scratch_size = self.unaligned_size(bx);
        let scratch_align = self.align(bx);
        let has_scratch = scratch_size > layout.size;
        let (store_dst, align) = if has_scratch {
            // We must allocate enough space for the final store instruction.
            let llscratch = bx.alloca(scratch_size, scratch_align);
            bx.lifetime_start(llscratch, scratch_size);
            (llscratch, scratch_align)
        } else {
            (dst, layout.align.abi)
        };
        match bx.type_kind(bx.val_ty(src)) {
            crate::common::TypeKind::Struct | crate::common::TypeKind::Array => {
                let mut index = 0;
                let mut offset = 0;
                for reg in self.prefix.iter().filter_map(|&x| x) {
                    let from = bx.extract_value(src, index);
                    let ptr = if index == 0 {
                        store_dst
                    } else {
                        bx.inbounds_ptradd(store_dst, bx.const_usize(offset))
                    };
                    bx.store(from, ptr, align);
                    index += 1;
                    offset += reg.size.bytes();
                }
                let (rest_count, rem_bytes) = if self.rest.unit.size.bytes() == 0 {
                    (0, 0)
                } else {
                    (
                        self.rest.total.bytes() / self.rest.unit.size.bytes(),
                        self.rest.total.bytes() % self.rest.unit.size.bytes(),
                    )
                };
                for _ in 0..rest_count {
                    let from = bx.extract_value(src, index);
                    let ptr = if offset == 0 {
                        store_dst
                    } else {
                        bx.inbounds_ptradd(store_dst, bx.const_usize(offset))
                    };
                    bx.store(from, ptr, align);
                    index += 1;
                    offset += self.rest.unit.size.bytes();
                }
                if rem_bytes != 0 {
                    let from = bx.extract_value(src, index);
                    let ptr = bx.inbounds_ptradd(store_dst, bx.const_usize(offset));
                    bx.store(from, ptr, align);
                }
            }
            ty_kind if bx.type_kind(bx.reg_backend_type(&self.rest.unit)) == ty_kind => {
                bx.store(src, store_dst, align);
            }
            ty_kind => bug!("cannot cast {ty_kind:?} to the Rust representation in CastTarget"),
        };
        if has_scratch {
            let tmp = bx.load(bx.backend_type(layout), store_dst, scratch_align);
            bx.lifetime_end(store_dst, scratch_size);
            bx.store(tmp, dst, layout.align.abi);
        }
    }
}
