//! Replaces calls to `Iter::next` with small, specialized MIR implementations, for some common iterators.
use rustc_abi::{FieldIdx, VariantIdx};
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{SourceInfo, *};
use rustc_middle::ty::{self, AdtDef, AdtKind, GenericArgs, Ty, TyCtxt};
use rustc_span::Span;
use rustc_type_ir::inherent::*;
use tracing::trace;

use crate::hir::def_id::{CrateNum, DefId};

pub(super) enum StreamlineIter {
    Working { core: CrateNum, iter_next: DefId },
    Disabled,
}
impl StreamlineIter {
    pub(crate) fn new(tcx: TyCtxt<'_>) -> Self {
        let Some(iter_next) = tcx.lang_items().next_fn() else {
            return Self::Disabled;
        };
        let core = iter_next.krate;
        Self::Working { core, iter_next }
    }
}
impl<'tcx> crate::MirPass<'tcx> for StreamlineIter {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 1 && (matches!(self, StreamlineIter::Working { .. }))
    }
    // Temporary allow for dev purposes
    #[allow(unused_variables, unused_mut, unreachable_code)]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running StreamlineIter on {:?}", body.source);
        let Self::Working { core, iter_next } = self else {
            return;
        };
        let mut bbs = body.basic_blocks.as_mut_preserves_cfg();
        let locals = &mut body.local_decls;
        // If any optimizations were pefromed, invalidate the cache.
        let mut cfg_invalid = false;

        // 1st. Go trough all terminators, find calls.
        for bid in (0..(bbs.len())).into_iter().map(BasicBlock::from_usize) {
            let mut bb = &bbs[bid];
            // Check if this is the call to std::slice::Iter::next OR std::slice::IterMut::next
            let Some(InlineSliceNextCandidate {
                iter_place,
                iter_adt,
                iter_args,
                fn_span,
                source_info,
                destination,
                target,
            }) = terminator_iter_next(&bb.terminator, *iter_next, *core, tcx)
            else {
                continue;
            };
            // Find the relevant field:
            let (notnull_idx, notnull_ty) = iter_adt
                .variant(VariantIdx::ZERO)
                .fields
                .iter()
                .enumerate()
                .map(|(idx, field)| (FieldIdx::from_usize(idx), field.ty(tcx, iter_args)))
                .filter(|(idx, ty)| match ty.kind() {
                    ty::Adt(adt, _) => !adt.is_phantom_data(),
                    _ => false,
                })
                .next()
                .unwrap();
            let iter_place = tcx.mk_place_deref(iter_place);
            let ptr_nonull = tcx.mk_place_field(iter_place, notnull_idx, notnull_ty);
            let ty::Adt(non_null_adt, on_null_arg) = notnull_ty.kind() else {
                continue;
            };
            let (inner_idx, inner_t) = non_null_adt
                .variant(VariantIdx::ZERO)
                .fields
                .iter()
                .enumerate()
                .map(|(idx, field)| (FieldIdx::from_usize(idx), field.ty(tcx, on_null_arg)))
                .filter(|(idx, ty)| match ty.kind() {
                    ty::RawPtr(_, _) => true,
                    _ => false,
                })
                .next()
                .unwrap();
            let pointer = tcx.mk_place_field(ptr_nonull, inner_idx, inner_t);
            // Increment pointer
            let val = Operand::Copy(pointer);
            let one = Operand::const_from_scalar(
                tcx,
                tcx.types.usize,
                Scalar::from_target_usize(1, &tcx),
                fn_span,
            );
            let offset = Rvalue::BinaryOp(BinOp::Offset, Box::new((val, one)));
            let incr =
                Statement { kind: StatementKind::Assign(Box::new((pointer, offset))), source_info };
            // Allocate the check & cast_end local:
            let check = locals.push(LocalDecl::new(tcx.types.bool, fn_span));
            // Bounds check
            let (idx, ty) = iter_adt
                .variant(VariantIdx::ZERO)
                .fields
                .iter()
                .enumerate()
                .map(|(idx, field)| (FieldIdx::from_usize(idx), field.ty(tcx, iter_args)))
                .filter(|(idx, ty)| match ty.kind() {
                    ty::RawPtr(_, _) => true,
                    _ => false,
                })
                .next()
                .unwrap();

            let end_ptr = tcx.mk_place_field(iter_place, idx, ty);
            let end_ptr = Operand::Copy(end_ptr);
            let ptr = Operand::Copy(pointer);
            let pointer_ty = pointer.ty(locals, tcx).ty;
            let end_ptr_after_cast = locals.push(LocalDecl::new(pointer_ty, fn_span));
            let cast_end_ptr = Rvalue::Cast(CastKind::PtrToPtr, end_ptr, pointer_ty);
            let ptr_cast = Statement {
                kind: StatementKind::Assign(Box::new((end_ptr_after_cast.into(), cast_end_ptr))),
                source_info,
            };

            let is_empty = Rvalue::BinaryOp(
                BinOp::Eq,
                Box::new((ptr, Operand::Copy(end_ptr_after_cast.into()))),
            );
            let check_iter_empty = Statement {
                kind: StatementKind::Assign(Box::new((check.into(), is_empty))),
                source_info,
            };

            // Create the Some and None blocks
            let rejoin = Terminator { kind: TerminatorKind::Goto { target }, source_info };
            let mut some_block = BasicBlockData::new(Some(rejoin.clone()), false);
            let mut none_block = BasicBlockData::new(Some(rejoin), false);
            // Create the None value
            let dst_ty = destination.ty(locals, tcx);
            let ty::Adt(option_adt, option_gargs) = dst_ty.ty.kind() else {
                continue;
            };
            let none_val = Rvalue::Aggregate(
                Box::new(AggregateKind::Adt(
                    option_adt.did(),
                    VariantIdx::ZERO,
                    option_gargs,
                    None,
                    None,
                )),
                IndexVec::new(),
            );
            let set_none = Statement {
                kind: StatementKind::Assign(Box::new((destination, none_val))),
                source_info,
            };
            none_block.statements.push(set_none);
            // Cast the pointer to a refernece, preserving lifetimes.
            let ref_ty = option_gargs[0].expect_ty();
            let ref_local = locals.push(LocalDecl::new(ref_ty, fn_span));

            let ty::Ref(region, _, muta) = ref_ty.kind() else {
                continue;
            };
            let pointer_local = locals.push(LocalDecl::new(pointer_ty, fn_span));
            let pointer_assign = Rvalue::Use(Operand::Copy(pointer));
            let pointer_assign = Statement {
                kind: StatementKind::Assign(Box::new((pointer_local.into(), pointer_assign))),
                source_info,
            };
            let borrow = if *muta == Mutability::Not {
                BorrowKind::Shared
            } else {
                BorrowKind::Mut { kind: MutBorrowKind::Default }
            };
            let rf = Rvalue::Ref(*region, borrow, tcx.mk_place_deref(pointer_local.into()));
            let rf = Statement {
                kind: StatementKind::Assign(Box::new((ref_local.into(), rf))),
                source_info,
            };
            let some_val = Rvalue::Aggregate(
                Box::new(AggregateKind::Adt(
                    option_adt.did(),
                    VariantIdx::from_usize(1),
                    option_gargs,
                    None,
                    None,
                )),
                [Operand::Move(ref_local.into())].into(),
            );
            let set_some = Statement {
                kind: StatementKind::Assign(Box::new((destination, some_val))),
                source_info,
            };
            some_block.statements.push(pointer_assign);
            some_block.statements.push(rf);
            some_block.statements.push(incr);
            some_block.statements.push(set_some);

            // Get the new blocks in place - this invalidates caches!
            cfg_invalid = true;
            let some_bb = bbs.push(some_block);
            let none_bb = bbs.push(none_block);

            // Change the original block.
            let mut bb = &mut bbs[bid];
            bb.terminator = Some(Terminator {
                kind: TerminatorKind::SwitchInt {
                    discr: Operand::Move(check.into()),
                    targets: SwitchTargets::new(std::iter::once((0, some_bb)), none_bb),
                },
                source_info,
            });
            bb.statements.push(ptr_cast);
            bb.statements.push(check_iter_empty);
        }
        if cfg_invalid {
            body.basic_blocks.invalidate_cfg_cache();
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}
fn not_zst<'tcx>(t: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> bool {
    match t.kind() {
        ty::Uint(_)
        | ty::Int(_)
        | ty::Bool
        | ty::Float(_)
        | ty::Char
        | ty::Ref(..)
        | ty::RawPtr(..)
        | ty::FnPtr(..) => true,
        ty::Tuple(elements) => elements.iter().any(|ty| not_zst(ty, tcx)),
        ty::Array(elem, count) if count.try_to_target_usize(tcx).is_some_and(|count| count > 0) => {
            not_zst(*elem, tcx)
        }
        ty::Array(_, _) => false,
        ty::Never | ty::FnDef(..) => false,
        ty::Adt(def, args) => match def.adt_kind() {
            AdtKind::Enum => def.variants().len() > 1,
            AdtKind::Struct | AdtKind::Union => def
                .variant(VariantIdx::ZERO)
                .fields
                .iter()
                .any(|field| not_zst(field.ty(tcx, args), tcx)),
        },
        // Generic's, can't determine if they are not-zst's at compile time.
        ty::Param(..) | ty::Alias(..) | ty::Bound(..) => false,
        // Those should not occur here, but I still handle them just in case.
        ty::Str | ty::Slice(..) | ty::Foreign(_) | ty::Dynamic(..) => false,
        ty::Pat(..) | ty::UnsafeBinder(..) | ty::Infer(..) | ty::Placeholder(_) | ty::Error(_) => {
            false
        }
        // There are ways to check if those are ZSTs, but this is not worth it ATM.
        ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Coroutine(..)
        | ty::CoroutineWitness(..) => false,
    }
}
//-Copt-level=3 -Zmir-opt-level=3 --emit=llvm-ir -C debug-assertions=no
struct InlineSliceNextCandidate<'tcx> {
    iter_place: Place<'tcx>,
    iter_adt: AdtDef<'tcx>,
    iter_args: &'tcx GenericArgs<'tcx>,
    fn_span: Span,
    source_info: SourceInfo,
    destination: Place<'tcx>,
    target: BasicBlock,
}
/// This function checks if this is a call to `std::slice::Iter::next` OR `std::slice::IterMut::next`.
/// Currently, it uses a bunch of ulgy things to do so, but if those iterators become lang items, then
/// this could be replaced by a simple DefID check.
#[allow(unreachable_code, unused_variables)]
fn terminator_iter_next<'tcx>(
    terminator: &Option<Terminator<'tcx>>,
    iter_next: DefId,
    core: CrateNum,
    tcx: TyCtxt<'tcx>,
) -> Option<InlineSliceNextCandidate<'tcx>> {
    use rustc_type_ir::inherent::*;
    let Terminator { kind, source_info } = terminator.as_ref()?;
    let TerminatorKind::Call {
        ref func,
        ref args,
        destination,
        target,
        unwind: _,
        call_source: _,
        fn_span,
    } = kind
    else {
        return None;
    };
    // 2. Check that the `func` of the call is known.
    let func = func.constant()?;
    // 3. Check that the `func` is FnDef
    let ty::FnDef(defid, generic_args) = func.ty().kind() else {
        return None;
    };
    // 4. Check that this is Iter::next
    if *defid != iter_next {
        return None;
    }
    // 5. Extract parts of the iterator
    let iter_ty = generic_args[0].expect_ty();
    let ty::Adt(iter_adt, iter_args) = iter_ty.kind() else {
        return None;
    };
    if iter_adt.did().krate != core {
        return None;
    }
    // 6. Check its argument count - this is a short, cheap check
    if iter_args.len() != 2 {
        return None;
    }
    // 7. Check that the first arg is a lifetime
    if iter_args[0].as_region().is_none() {
        return None;
    }
    // 8. Check that this ADT is a struct, and has 3 fields.
    if !iter_adt.is_struct() {
        return None;
    }
    if iter_adt.all_fields().count() != 3 {
        return None;
    }
    // Check that it has a *const T field.
    if !iter_adt.all_field_tys(tcx).skip_binder().into_iter().any(|ty| match ty.kind() {
        ty::RawPtr(_, _) => true,
        _ => false,
    }) {
        return None;
    }
    // 7. Check that the name of this ADT is `slice::iter::Iter`. This is a janky way to check if this is the iterator we are interested in.
    let name = format!("{:?}", iter_adt.did());
    if !name.as_str().contains("slice::iter::Iter") {
        return None;
    }
    // We now know this is a slice iterator - so we can optimize it !
    // Check if we know if this is not a `zst`
    if !not_zst(iter_args[1].expect_ty(), tcx) {
        return None;
    }

    // We found `slice::iter::Iter`, now, we can work on optimizing it away.
    // 1. Get the `ptr.pointer` field - this is the field we will increment.
    // We know that Iter::next() takes a &mut self, which can't be a constant(?). So, we only worry about Operand::Move or Operand::Copy, which can be turned into places.
    let Some(iter_place) = args[0].node.place() else {
        return None;
    };
    Some(InlineSliceNextCandidate {
        iter_place,
        iter_adt: *iter_adt,
        iter_args,
        fn_span: *fn_span,
        source_info: *source_info,
        destination: *destination,
        target: target.as_ref().copied()?,
    })
}
