use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::mir::*;
use rustc::ty::{self, Predicate, Ty, TyCtxt, adjustment::{PointerCast}};
use rustc_target::spec::abi;
use std::borrow::Cow;
use syntax_pos::Span;

type McfResult = Result<(), (Span, Cow<'static, str>)>;

pub fn is_min_const_fn(tcx: TyCtxt<'tcx>, def_id: DefId, body: &'a Body<'tcx>) -> McfResult {
    let mut current = def_id;
    loop {
        let predicates = tcx.predicates_of(current);
        for (predicate, _) in &predicates.predicates {
            match predicate {
                | Predicate::RegionOutlives(_)
                | Predicate::TypeOutlives(_)
                | Predicate::WellFormed(_)
                | Predicate::Projection(_)
                | Predicate::ConstEvaluatable(..) => continue,
                | Predicate::ObjectSafe(_) => {
                    bug!("object safe predicate on function: {:#?}", predicate)
                }
                Predicate::ClosureKind(..) => {
                    bug!("closure kind predicate on function: {:#?}", predicate)
                }
                Predicate::Subtype(_) => bug!("subtype predicate on function: {:#?}", predicate),
                Predicate::Trait(pred) => {
                    if Some(pred.def_id()) == tcx.lang_items().sized_trait() {
                        continue;
                    }
                    match pred.skip_binder().self_ty().sty {
                        ty::Param(ref p) => {
                            let generics = tcx.generics_of(current);
                            let def = generics.type_param(p, tcx);
                            let span = tcx.def_span(def.def_id);
                            return Err((
                                span,
                                "trait bounds other than `Sized` \
                                 on const fn parameters are unstable"
                                    .into(),
                            ));
                        }
                        // other kinds of bounds are either tautologies
                        // or cause errors in other passes
                        _ => continue,
                    }
                }
            }
        }
        match predicates.parent {
            Some(parent) => current = parent,
            None => break,
        }
    }

    for local in &body.local_decls {
        check_ty(tcx, local.ty, local.source_info.span, def_id)?;
    }
    // impl trait is gone in MIR, so check the return type manually
    check_ty(
        tcx,
        tcx.fn_sig(def_id).output().skip_binder(),
        body.local_decls.iter().next().unwrap().source_info.span,
        def_id,
    )?;

    for bb in body.basic_blocks() {
        check_terminator(tcx, body, bb.terminator())?;
        for stmt in &bb.statements {
            check_statement(tcx, body, stmt)?;
        }
    }
    Ok(())
}

fn check_ty(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span, fn_def_id: DefId) -> McfResult {
    for ty in ty.walk() {
        match ty.sty {
            ty::Ref(_, _, hir::Mutability::MutMutable) => return Err((
                span,
                "mutable references in const fn are unstable".into(),
            )),
            ty::Opaque(..) => return Err((span, "`impl Trait` in const fn is unstable".into())),
            ty::FnPtr(..) => {
                if !tcx.const_fn_is_allowed_fn_ptr(fn_def_id) {
                    return Err((span, "function pointers in const fn are unstable".into()))
                }
            }
            ty::Dynamic(preds, _) => {
                for pred in preds.iter() {
                    match pred.skip_binder() {
                        | ty::ExistentialPredicate::AutoTrait(_)
                        | ty::ExistentialPredicate::Projection(_) => {
                            return Err((
                                span,
                                "trait bounds other than `Sized` \
                                 on const fn parameters are unstable"
                                    .into(),
                            ))
                        }
                        ty::ExistentialPredicate::Trait(trait_ref) => {
                            if Some(trait_ref.def_id) != tcx.lang_items().sized_trait() {
                                return Err((
                                    span,
                                    "trait bounds other than `Sized` \
                                     on const fn parameters are unstable"
                                        .into(),
                                ));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_rvalue(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    rvalue: &Rvalue<'tcx>,
    span: Span,
) -> McfResult {
    match rvalue {
        Rvalue::Repeat(operand, _) | Rvalue::Use(operand) => {
            check_operand(operand, span)
        }
        Rvalue::Len(place) | Rvalue::Discriminant(place) | Rvalue::Ref(_, _, place) => {
            check_place(place, span)
        }
        Rvalue::Cast(CastKind::Misc, operand, cast_ty) => {
            use rustc::ty::cast::CastTy;
            let cast_in = CastTy::from_ty(operand.ty(body, tcx)).expect("bad input type for cast");
            let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
            match (cast_in, cast_out) {
                (CastTy::Ptr(_), CastTy::Int(_)) | (CastTy::FnPtr, CastTy::Int(_)) => Err((
                    span,
                    "casting pointers to ints is unstable in const fn".into(),
                )),
                (CastTy::RPtr(_), CastTy::Float) => bug!(),
                (CastTy::RPtr(_), CastTy::Int(_)) => bug!(),
                (CastTy::Ptr(_), CastTy::RPtr(_)) => bug!(),
                _ => check_operand(operand, span),
            }
        }
        Rvalue::Cast(CastKind::Pointer(PointerCast::MutToConstPointer), operand, _) => {
            check_operand(operand, span)
        }
        Rvalue::Cast(CastKind::Pointer(PointerCast::UnsafeFnPointer), _, _) |
        Rvalue::Cast(CastKind::Pointer(PointerCast::ClosureFnPointer(_)), _, _) |
        Rvalue::Cast(CastKind::Pointer(PointerCast::ReifyFnPointer), _, _) => Err((
            span,
            "function pointer casts are not allowed in const fn".into(),
        )),
        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), _, _) => Err((
            span,
            "unsizing casts are not allowed in const fn".into(),
        )),
        // binops are fine on integers
        Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
            check_operand(lhs, span)?;
            check_operand(rhs, span)?;
            let ty = lhs.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() || ty.is_char() {
                Ok(())
            } else {
                Err((
                    span,
                    "only int, `bool` and `char` operations are stable in const fn".into(),
                ))
            }
        }
        Rvalue::NullaryOp(NullOp::SizeOf, _) => Ok(()),
        Rvalue::NullaryOp(NullOp::Box, _) => Err((
            span,
            "heap allocations are not allowed in const fn".into(),
        )),
        Rvalue::UnaryOp(_, operand) => {
            let ty = operand.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() {
                check_operand(operand, span)
            } else {
                Err((
                    span,
                    "only int and `bool` operations are stable in const fn".into(),
                ))
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                check_operand(operand, span)?;
            }
            Ok(())
        }
    }
}

fn check_statement(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    statement: &Statement<'tcx>,
) -> McfResult {
    let span = statement.source_info.span;
    match &statement.kind {
        StatementKind::Assign(place, rval) => {
            check_place(place, span)?;
            check_rvalue(tcx, body, rval, span)
        }

        StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _) => {
            Err((span, "loops and conditional expressions are not stable in const fn".into()))
        }

        StatementKind::FakeRead(_, place) => check_place(place, span),

        // just an assignment
        StatementKind::SetDiscriminant { .. } => Ok(()),

        | StatementKind::InlineAsm { .. } => {
            Err((span, "cannot use inline assembly in const fn".into()))
        }

        // These are all NOPs
        | StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..)
        | StatementKind::Nop => Ok(()),
    }
}

fn check_operand(
    operand: &Operand<'tcx>,
    span: Span,
) -> McfResult {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => {
            check_place(place, span)
        }
        Operand::Constant(_) => Ok(()),
    }
}

fn check_place(
    place: &Place<'tcx>,
    span: Span,
) -> McfResult {
    place.iterate(|place_base, place_projection| {
        for proj in place_projection {
            match proj.elem {
                ProjectionElem::Downcast(..) => {
                    return Err((span, "`match` or `if let` in `const fn` is unstable".into()));
                }
                ProjectionElem::ConstantIndex { .. }
                | ProjectionElem::Subslice { .. }
                | ProjectionElem::Deref
                | ProjectionElem::Field(..)
                | ProjectionElem::Index(_) => {}
            }
        }

        match place_base {
            PlaceBase::Static(box Static { kind: StaticKind::Static(_), .. }) => {
                Err((span, "cannot access `static` items in const fn".into()))
            }
            PlaceBase::Local(_)
            | PlaceBase::Static(box Static { kind: StaticKind::Promoted(_), .. }) => Ok(()),
        }
    })
}

fn check_terminator(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    terminator: &Terminator<'tcx>,
) -> McfResult {
    let span = terminator.source_info.span;
    match &terminator.kind {
        | TerminatorKind::Goto { .. }
        | TerminatorKind::Return
        | TerminatorKind::Resume => Ok(()),

        TerminatorKind::Drop { location, .. } => {
            check_place(location, span)
        }
        TerminatorKind::DropAndReplace { location, value, .. } => {
            check_place(location, span)?;
            check_operand(value, span)
        },

        TerminatorKind::FalseEdges { .. } | TerminatorKind::SwitchInt { .. } => Err((
            span,
            "loops and conditional expressions are not stable in const fn".into(),
        )),
        | TerminatorKind::Abort | TerminatorKind::Unreachable => {
            Err((span, "const fn with unreachable code is not stable".into()))
        }
        | TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => {
            Err((span, "const fn generators are unstable".into()))
        }

        TerminatorKind::Call {
            func,
            args,
            from_hir_call: _,
            destination: _,
            cleanup: _,
        } => {
            let fn_ty = func.ty(body, tcx);
            if let ty::FnDef(def_id, _) = fn_ty.sty {

                // some intrinsics are waved through if called inside the
                // standard library. Users never need to call them directly
                match tcx.fn_sig(def_id).abi() {
                    abi::Abi::RustIntrinsic => if !is_intrinsic_whitelisted(tcx, def_id) {
                        return Err((
                            span,
                            "can only call a curated list of intrinsics in `min_const_fn`".into(),
                        ))
                    },
                    abi::Abi::Rust if tcx.is_min_const_fn(def_id) => {},
                    abi::Abi::Rust => return Err((
                        span,
                        format!(
                            "can only call other `const fn` within a `const fn`, \
                             but `{:?}` is not stable as `const fn`",
                            func,
                        )
                        .into(),
                    )),
                    abi => return Err((
                        span,
                        format!(
                            "cannot call functions with `{}` abi in `min_const_fn`",
                            abi,
                        ).into(),
                    )),
                }

                check_operand(func, span)?;

                for arg in args {
                    check_operand(arg, span)?;
                }
                Ok(())
            } else {
                Err((span, "can only call other const fns within const fn".into()))
            }
        }

        TerminatorKind::Assert {
            cond,
            expected: _,
            msg: _,
            target: _,
            cleanup: _,
        } => check_operand(cond, span),

        TerminatorKind::FalseUnwind { .. } => {
            Err((span, "loops are not allowed in const fn".into()))
        },
    }
}

/// Returns `true` if the `def_id` refers to an intrisic which we've whitelisted
/// for being called from stable `const fn`s (`min_const_fn`).
///
/// Adding more intrinsics requires sign-off from @rust-lang/lang.
fn is_intrinsic_whitelisted(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    match &tcx.item_name(def_id).as_str()[..] {
        | "size_of"
        | "min_align_of"
        | "needs_drop"
        // Arithmetic:
        | "add_with_overflow" // ~> .overflowing_add
        | "sub_with_overflow" // ~> .overflowing_sub
        | "mul_with_overflow" // ~> .overflowing_mul
        | "overflowing_add" // ~> .wrapping_add
        | "overflowing_sub" // ~> .wrapping_sub
        | "overflowing_mul" // ~> .wrapping_mul
        | "saturating_add" // ~> .saturating_add
        | "saturating_sub" // ~> .saturating_sub
        | "unchecked_shl" // ~> .wrapping_shl
        | "unchecked_shr" // ~> .wrapping_shr
        | "rotate_left" // ~> .rotate_left
        | "rotate_right" // ~> .rotate_right
        | "ctpop" // ~> .count_ones
        | "ctlz" // ~> .leading_zeros
        | "cttz" // ~> .trailing_zeros
        | "bswap" // ~> .swap_bytes
        | "bitreverse" // ~> .reverse_bits
        => true,
        _ => false,
    }
}
