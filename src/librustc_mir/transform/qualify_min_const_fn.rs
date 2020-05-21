use rustc_attr as attr;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::*;
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, adjustment::PointerCast, Ty, TyCtxt};
use rustc_span::symbol::{sym, Symbol};
use rustc_span::Span;
use std::borrow::Cow;

type McfResult = Result<(), (Span, Cow<'static, str>)>;

pub fn is_min_const_fn(tcx: TyCtxt<'tcx>, def_id: DefId, body: &'a Body<'tcx>) -> McfResult {
    // Prevent const trait methods from being annotated as `stable`.
    if tcx.features().staged_api {
        let hir_id = tcx.hir().as_local_hir_id(def_id.expect_local());
        if crate::const_eval::is_parent_const_impl_raw(tcx, hir_id) {
            return Err((body.span, "trait methods cannot be stable const fn".into()));
        }
    }

    let mut current = def_id;
    loop {
        let predicates = tcx.predicates_of(current);
        for (predicate, _) in predicates.predicates {
            match predicate.kind() {
                ty::PredicateKind::RegionOutlives(_)
                | ty::PredicateKind::TypeOutlives(_)
                | ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::Projection(_)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..) => continue,
                ty::PredicateKind::ObjectSafe(_) => {
                    bug!("object safe predicate on function: {:#?}", predicate)
                }
                ty::PredicateKind::ClosureKind(..) => {
                    bug!("closure kind predicate on function: {:#?}", predicate)
                }
                ty::PredicateKind::Subtype(_) => {
                    bug!("subtype predicate on function: {:#?}", predicate)
                }
                &ty::PredicateKind::Trait(pred, constness) => {
                    if Some(pred.def_id()) == tcx.lang_items().sized_trait() {
                        continue;
                    }
                    match pred.skip_binder().self_ty().kind {
                        ty::Param(ref p) => {
                            // Allow `T: ?const Trait`
                            if constness == hir::Constness::NotConst
                                && feature_allowed(tcx, def_id, sym::const_trait_bound_opt_out)
                            {
                                continue;
                            }

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
        check_terminator(tcx, body, def_id, bb.terminator())?;
        for stmt in &bb.statements {
            check_statement(tcx, body, def_id, stmt)?;
        }
    }
    Ok(())
}

fn check_ty(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span, fn_def_id: DefId) -> McfResult {
    for arg in ty.walk() {
        let ty = match arg.unpack() {
            GenericArgKind::Type(ty) => ty,

            // No constraints on lifetimes or constants, except potentially
            // constants' types, but `walk` will get to them as well.
            GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => continue,
        };

        match ty.kind {
            ty::Ref(_, _, hir::Mutability::Mut) => {
                if !feature_allowed(tcx, fn_def_id, sym::const_mut_refs) {
                    return Err((span, "mutable references in const fn are unstable".into()));
                }
            }
            ty::Opaque(..) => return Err((span, "`impl Trait` in const fn is unstable".into())),
            ty::FnPtr(..) => {
                if !tcx.const_fn_is_allowed_fn_ptr(fn_def_id) {
                    return Err((span, "function pointers in const fn are unstable".into()));
                }
            }
            ty::Dynamic(preds, _) => {
                for pred in preds.iter() {
                    match pred.skip_binder() {
                        ty::ExistentialPredicate::AutoTrait(_)
                        | ty::ExistentialPredicate::Projection(_) => {
                            return Err((
                                span,
                                "trait bounds other than `Sized` \
                                 on const fn parameters are unstable"
                                    .into(),
                            ));
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
    body: &Body<'tcx>,
    def_id: DefId,
    rvalue: &Rvalue<'tcx>,
    span: Span,
) -> McfResult {
    match rvalue {
        Rvalue::Repeat(operand, _) | Rvalue::Use(operand) => {
            check_operand(tcx, operand, span, def_id, body)
        }
        Rvalue::Len(place)
        | Rvalue::Discriminant(place)
        | Rvalue::Ref(_, _, place)
        | Rvalue::AddressOf(_, place) => check_place(tcx, *place, span, def_id, body),
        Rvalue::Cast(CastKind::Misc, operand, cast_ty) => {
            use rustc_middle::ty::cast::CastTy;
            let cast_in = CastTy::from_ty(operand.ty(body, tcx)).expect("bad input type for cast");
            let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
            match (cast_in, cast_out) {
                (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) => {
                    Err((span, "casting pointers to ints is unstable in const fn".into()))
                }
                _ => check_operand(tcx, operand, span, def_id, body),
            }
        }
        Rvalue::Cast(
            CastKind::Pointer(PointerCast::MutToConstPointer | PointerCast::ArrayToPointer),
            operand,
            _,
        ) => check_operand(tcx, operand, span, def_id, body),
        Rvalue::Cast(
            CastKind::Pointer(
                PointerCast::UnsafeFnPointer
                | PointerCast::ClosureFnPointer(_)
                | PointerCast::ReifyFnPointer,
            ),
            _,
            _,
        ) => Err((span, "function pointer casts are not allowed in const fn".into())),
        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), _, _) => {
            Err((span, "unsizing casts are not allowed in const fn".into()))
        }
        // binops are fine on integers
        Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
            check_operand(tcx, lhs, span, def_id, body)?;
            check_operand(tcx, rhs, span, def_id, body)?;
            let ty = lhs.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() || ty.is_char() {
                Ok(())
            } else {
                Err((span, "only int, `bool` and `char` operations are stable in const fn".into()))
            }
        }
        Rvalue::NullaryOp(NullOp::SizeOf, _) => Ok(()),
        Rvalue::NullaryOp(NullOp::Box, _) => {
            Err((span, "heap allocations are not allowed in const fn".into()))
        }
        Rvalue::UnaryOp(_, operand) => {
            let ty = operand.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() {
                check_operand(tcx, operand, span, def_id, body)
            } else {
                Err((span, "only int and `bool` operations are stable in const fn".into()))
            }
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                check_operand(tcx, operand, span, def_id, body)?;
            }
            Ok(())
        }
    }
}

fn check_statement(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    statement: &Statement<'tcx>,
) -> McfResult {
    let span = statement.source_info.span;
    match &statement.kind {
        StatementKind::Assign(box (place, rval)) => {
            check_place(tcx, *place, span, def_id, body)?;
            check_rvalue(tcx, body, def_id, rval, span)
        }

        StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _)
            if !feature_allowed(tcx, def_id, sym::const_if_match) =>
        {
            Err((span, "loops and conditional expressions are not stable in const fn".into()))
        }

        StatementKind::FakeRead(_, place) => check_place(tcx, **place, span, def_id, body),

        // just an assignment
        StatementKind::SetDiscriminant { place, .. } => {
            check_place(tcx, **place, span, def_id, body)
        }

        StatementKind::LlvmInlineAsm { .. } => {
            Err((span, "cannot use inline assembly in const fn".into()))
        }

        // These are all NOPs
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..)
        | StatementKind::Nop => Ok(()),
    }
}

fn check_operand(
    tcx: TyCtxt<'tcx>,
    operand: &Operand<'tcx>,
    span: Span,
    def_id: DefId,
    body: &Body<'tcx>,
) -> McfResult {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => check_place(tcx, *place, span, def_id, body),
        Operand::Constant(c) => match c.check_static_ptr(tcx) {
            Some(_) => Err((span, "cannot access `static` items in const fn".into())),
            None => Ok(()),
        },
    }
}

fn check_place(
    tcx: TyCtxt<'tcx>,
    place: Place<'tcx>,
    span: Span,
    def_id: DefId,
    body: &Body<'tcx>,
) -> McfResult {
    let mut cursor = place.projection.as_ref();
    while let &[ref proj_base @ .., elem] = cursor {
        cursor = proj_base;
        match elem {
            ProjectionElem::Field(..) => {
                let base_ty = Place::ty_from(place.local, &proj_base, body, tcx).ty;
                if let Some(def) = base_ty.ty_adt_def() {
                    // No union field accesses in `const fn`
                    if def.is_union() {
                        if !feature_allowed(tcx, def_id, sym::const_fn_union) {
                            return Err((span, "accessing union fields is unstable".into()));
                        }
                    }
                }
            }
            ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::Downcast(..)
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Deref
            | ProjectionElem::Index(_) => {}
        }
    }

    Ok(())
}

/// Returns `true` if the given feature gate is allowed within the function with the given `DefId`.
fn feature_allowed(tcx: TyCtxt<'tcx>, def_id: DefId, feature_gate: Symbol) -> bool {
    // All features require that the corresponding gate be enabled,
    // even if the function has `#[allow_internal_unstable(the_gate)]`.
    if !tcx.features().enabled(feature_gate) {
        return false;
    }

    // If this crate is not using stability attributes, or this function is not claiming to be a
    // stable `const fn`, that is all that is required.
    if !tcx.features().staged_api || tcx.has_attr(def_id, sym::rustc_const_unstable) {
        return true;
    }

    // However, we cannot allow stable `const fn`s to use unstable features without an explicit
    // opt-in via `allow_internal_unstable`.
    attr::allow_internal_unstable(&tcx.get_attrs(def_id), &tcx.sess.diagnostic())
        .map_or(false, |mut features| features.any(|name| name == feature_gate))
}

fn check_terminator(
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    def_id: DefId,
    terminator: &Terminator<'tcx>,
) -> McfResult {
    let span = terminator.source_info.span;
    match &terminator.kind {
        TerminatorKind::FalseEdges { .. }
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::Goto { .. }
        | TerminatorKind::Return
        | TerminatorKind::Resume
        | TerminatorKind::Unreachable => Ok(()),

        TerminatorKind::Drop { location, .. } => check_place(tcx, *location, span, def_id, body),
        TerminatorKind::DropAndReplace { location, value, .. } => {
            check_place(tcx, *location, span, def_id, body)?;
            check_operand(tcx, value, span, def_id, body)
        }

        TerminatorKind::SwitchInt { .. } if !feature_allowed(tcx, def_id, sym::const_if_match) => {
            Err((span, "loops and conditional expressions are not stable in const fn".into()))
        }

        TerminatorKind::SwitchInt { discr, switch_ty: _, values: _, targets: _ } => {
            check_operand(tcx, discr, span, def_id, body)
        }

        TerminatorKind::Abort => Err((span, "abort is not stable in const fn".into())),
        TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => {
            Err((span, "const fn generators are unstable".into()))
        }

        TerminatorKind::Call { func, args, from_hir_call: _, destination: _, cleanup: _ } => {
            let fn_ty = func.ty(body, tcx);
            if let ty::FnDef(def_id, _) = fn_ty.kind {
                if !crate::const_eval::is_min_const_fn(tcx, def_id) {
                    return Err((
                        span,
                        format!(
                            "can only call other `const fn` within a `const fn`, \
                             but `{:?}` is not stable as `const fn`",
                            func,
                        )
                        .into(),
                    ));
                }

                check_operand(tcx, func, span, def_id, body)?;

                for arg in args {
                    check_operand(tcx, arg, span, def_id, body)?;
                }
                Ok(())
            } else {
                Err((span, "can only call other const fns within const fn".into()))
            }
        }

        TerminatorKind::Assert { cond, expected: _, msg: _, target: _, cleanup: _ } => {
            check_operand(tcx, cond, span, def_id, body)
        }

        TerminatorKind::InlineAsm { .. } => {
            Err((span, "cannot use inline assembly in const fn".into()))
        }
    }
}
