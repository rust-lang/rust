use rustc_abi::BackendRepr;
use rustc_hir::LangItem;
use rustc_middle::mir;
use rustc_middle::mir::visit::{NonMutatingUseContext, PlaceContext, Visitor};
use rustc_middle::ty::{Mutability, Ty, TyCtxt};
use rustc_span::Span;
use rustc_span::def_id::LOCAL_CRATE;
use rustc_target::abi::{Float, Integer, Niche, Primitive, Scalar, Size, WrappingRange};
use tracing::instrument;

use super::FunctionCx;
use crate::mir::OperandValue;
use crate::mir::place::PlaceValue;
use crate::traits::*;
use crate::{base, common};

pub(super) struct NicheFinder<'s, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> {
    pub(super) fx: &'s mut FunctionCx<'a, 'tcx, Bx>,
    pub(super) bx: &'s mut Bx,
    pub(super) places: Vec<(mir::Operand<'tcx>, Niche)>,
}

impl<'s, 'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> Visitor<'tcx> for NicheFinder<'s, 'a, 'tcx, Bx> {
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: mir::Location) {
        match rvalue {
            mir::Rvalue::Cast(mir::CastKind::Transmute, op, ty) => {
                let ty = self.fx.monomorphize(*ty);
                if let Some(niche) = self.bx.layout_of(ty).largest_niche {
                    self.places.push((op.clone(), niche));
                }
            }
            _ => self.super_rvalue(rvalue, location),
        }
    }

    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, _location: mir::Location) {
        if let mir::TerminatorKind::Return = terminator.kind {
            let op = mir::Operand::Copy(mir::Place::return_place());
            let ty = op.ty(self.fx.mir, self.bx.tcx());
            let ty = self.fx.monomorphize(ty);
            if let Some(niche) = self.bx.layout_of(ty).largest_niche {
                self.places.push((op, niche));
            }
        }
    }

    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: PlaceContext,
        _location: mir::Location,
    ) {
        match context {
            PlaceContext::NonMutatingUse(
                NonMutatingUseContext::Copy | NonMutatingUseContext::Move,
            ) => {}
            _ => {
                return;
            }
        }

        let ty = place.ty(self.fx.mir, self.bx.tcx()).ty;
        let ty = self.fx.monomorphize(ty);
        if let Some(niche) = self.bx.layout_of(ty).largest_niche {
            self.places.push((mir::Operand::Copy(*place), niche));
        };
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    fn value_in_niche(
        &mut self,
        bx: &mut Bx,
        op: crate::mir::OperandRef<'tcx, Bx::Value>,
        niche: Niche,
    ) -> Option<Bx::Value> {
        let niche_ty = niche.ty(bx.tcx());
        let niche_layout = bx.layout_of(niche_ty);

        let (imm, from_scalar, from_backend_ty) = match op.val {
            OperandValue::Immediate(imm) => {
                let BackendRepr::Scalar(from_scalar) = op.layout.backend_repr else {
                    unreachable!()
                };
                let from_backend_ty = bx.backend_type(op.layout);
                (imm, from_scalar, from_backend_ty)
            }
            OperandValue::Pair(first, second) => {
                let BackendRepr::ScalarPair(first_scalar, second_scalar) = op.layout.backend_repr
                else {
                    unreachable!()
                };
                if niche.offset == Size::ZERO {
                    (first, first_scalar, bx.scalar_pair_element_backend_type(op.layout, 0, true))
                } else {
                    // yolo
                    (second, second_scalar, bx.scalar_pair_element_backend_type(op.layout, 1, true))
                }
            }
            OperandValue::ZeroSized => unreachable!(),
            OperandValue::Ref(PlaceValue { llval: ptr, .. }) => {
                // General case: Load the niche primitive via pointer arithmetic.
                let niche_ptr_ty = Ty::new_ptr(bx.tcx(), niche_ty, Mutability::Not);
                let ptr = bx.pointercast(ptr, bx.backend_type(bx.layout_of(niche_ptr_ty)));

                let offset = niche.offset.bytes() / niche_layout.size.bytes();
                let niche_backend_ty = bx.backend_type(bx.layout_of(niche_ty));
                let ptr = bx.inbounds_gep(niche_backend_ty, ptr, &[bx.const_usize(offset)]);
                let value = bx.load(niche_backend_ty, ptr, rustc_target::abi::Align::ONE);
                return Some(value);
            }
        };

        // Any type whose ABI is a Scalar bool is turned into an i1, so it cannot contain a value
        // outside of its niche.
        if from_scalar.is_bool() {
            return None;
        }

        let to_scalar = Scalar::Initialized {
            value: niche.value,
            valid_range: WrappingRange::full(niche.size(bx.tcx())),
        };
        let to_backend_ty = bx.backend_type(niche_layout);
        if from_backend_ty == to_backend_ty {
            return Some(imm);
        }
        let value = self.transmute_immediate(
            bx,
            imm,
            from_scalar,
            from_backend_ty,
            to_scalar,
            to_backend_ty,
        );
        Some(value)
    }

    #[instrument(level = "debug", skip(self, bx))]
    pub fn codegen_niche_check(
        &mut self,
        bx: &mut Bx,
        mir_op: mir::Operand<'tcx>,
        niche: Niche,
        source_info: mir::SourceInfo,
    ) {
        let tcx = bx.tcx();
        let op_ty = self.monomorphize(mir_op.ty(self.mir, tcx));
        if op_ty == tcx.types.bool {
            return;
        }

        let op = self.codegen_operand(bx, &mir_op);

        let Some(value_in_niche) = self.value_in_niche(bx, op, niche) else {
            return;
        };
        let size = niche.size(tcx);

        let start = niche.scalar(niche.valid_range.start, bx);
        let end = niche.scalar(niche.valid_range.end, bx);

        let binop_le = base::bin_op_to_icmp_predicate(mir::BinOp::Le, false);
        let binop_ge = base::bin_op_to_icmp_predicate(mir::BinOp::Ge, false);
        let is_valid = if niche.valid_range.start == 0 {
            bx.icmp(binop_le, value_in_niche, end)
        } else if niche.valid_range.end == (u128::MAX >> 128 - size.bits()) {
            bx.icmp(binop_ge, value_in_niche, start)
        } else {
            // We need to check if the value is within a *wrapping* range. We could do this:
            // (niche >= start) && (niche <= end)
            // But what we're going to actually do is this:
            // max = end - start
            // (niche - start) <= max
            // The latter is much more complicated conceptually, but is actually less operations
            // because we can compute max in codegen.
            let mut max = niche.valid_range.end.wrapping_sub(niche.valid_range.start);
            let size = niche.size(tcx);
            if size.bits() < 128 {
                let mask = (1 << size.bits()) - 1;
                max &= mask;
            }
            let max_adjusted_allowed_value = niche.scalar(max, bx);

            let biased = bx.sub(value_in_niche, start);
            bx.icmp(binop_le, biased, max_adjusted_allowed_value)
        };

        // Create destination blocks, branching on is_valid
        let panic = bx.append_sibling_block("panic");
        let success = bx.append_sibling_block("success");
        bx.cond_br(is_valid, success, panic);

        // Switch to the failure block and codegen a call to the panic intrinsic
        bx.switch_to_block(panic);
        self.set_debug_loc(bx, source_info);
        let location = self.get_caller_location(bx, source_info).immediate();
        self.codegen_panic(
            bx,
            niche.lang_item(),
            &[value_in_niche, start, end, location],
            source_info.span,
        );

        // Continue codegen in the success block.
        bx.switch_to_block(success);
        self.set_debug_loc(bx, source_info);
    }

    #[instrument(level = "debug", skip(self, bx))]
    fn codegen_panic(&mut self, bx: &mut Bx, lang_item: LangItem, args: &[Bx::Value], span: Span) {
        if bx.tcx().is_compiler_builtins(LOCAL_CRATE) {
            bx.abort()
        } else {
            let (fn_abi, fn_ptr, instance) = common::build_langcall(bx, Some(span), lang_item);
            let fn_ty = bx.fn_decl_backend_type(&fn_abi);
            let fn_attrs = if bx.tcx().def_kind(self.instance.def_id()).has_codegen_attrs() {
                Some(bx.tcx().codegen_fn_attrs(self.instance.def_id()))
            } else {
                None
            };
            bx.call(fn_ty, fn_attrs, Some(&fn_abi), fn_ptr, args, None, Some(instance));
        }
        bx.unreachable();
    }
}

trait NicheExt {
    fn ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx>;
    fn size<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Size;
    fn scalar<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(&self, val: u128, bx: &mut Bx) -> Bx::Value;
    fn lang_item(&self) -> LangItem;
}

impl NicheExt for Niche {
    fn lang_item(&self) -> LangItem {
        match self.value {
            Primitive::Int(Integer::I8, _) => LangItem::PanicOccupiedNicheU8,
            Primitive::Int(Integer::I16, _) => LangItem::PanicOccupiedNicheU16,
            Primitive::Int(Integer::I32, _) => LangItem::PanicOccupiedNicheU32,
            Primitive::Int(Integer::I64, _) => LangItem::PanicOccupiedNicheU64,
            Primitive::Int(Integer::I128, _) => LangItem::PanicOccupiedNicheU128,
            Primitive::Pointer(_) => LangItem::PanicOccupiedNichePtr,
            Primitive::Float(Float::F16) => LangItem::PanicOccupiedNicheU16,
            Primitive::Float(Float::F32) => LangItem::PanicOccupiedNicheU32,
            Primitive::Float(Float::F64) => LangItem::PanicOccupiedNicheU64,
            Primitive::Float(Float::F128) => LangItem::PanicOccupiedNicheU128,
        }
    }

    fn ty<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        let types = &tcx.types;
        match self.value {
            Primitive::Int(Integer::I8, _) => types.u8,
            Primitive::Int(Integer::I16, _) => types.u16,
            Primitive::Int(Integer::I32, _) => types.u32,
            Primitive::Int(Integer::I64, _) => types.u64,
            Primitive::Int(Integer::I128, _) => types.u128,
            Primitive::Pointer(_) => Ty::new_ptr(tcx, types.unit, Mutability::Not),
            Primitive::Float(Float::F16) => types.u16,
            Primitive::Float(Float::F32) => types.u32,
            Primitive::Float(Float::F64) => types.u64,
            Primitive::Float(Float::F128) => types.u128,
        }
    }

    fn size<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Size {
        self.value.size(&tcx)
    }

    fn scalar<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(&self, val: u128, bx: &mut Bx) -> Bx::Value {
        use rustc_middle::mir::interpret::{Pointer, Scalar};
        let tcx = bx.tcx();
        let niche_ty = self.ty(tcx);
        let value = if niche_ty.is_any_ptr() {
            Scalar::from_maybe_pointer(Pointer::from_addr_invalid(val as u64), &tcx)
        } else {
            Scalar::from_uint(val, self.size(tcx))
        };
        let layout = rustc_target::abi::Scalar::Initialized {
            value: self.value,
            valid_range: WrappingRange::full(self.size(tcx)),
        };
        bx.scalar_to_backend(value, layout, bx.backend_type(bx.layout_of(self.ty(tcx))))
    }
}
