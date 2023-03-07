// This code used to be a part of `rustc` but moved to Clippy as a result of
// https://github.com/rust-lang/rust/issues/76618. Because of that, it contains unused code and some
// of terminologies might not be relevant in the context of Clippy. Note that its behavior might
// differ from the time of `rustc` even if the name stays the same.

use crate::msrvs::Msrv;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{
    Body, CastKind, NonDivergingIntrinsic, NullOp, Operand, Place, ProjectionElem, Rvalue, Statement, StatementKind,
    Terminator, TerminatorKind,
};
use rustc_middle::ty::subst::GenericArgKind;
use rustc_middle::ty::{self, adjustment::PointerCast, Ty, TyCtxt};
use rustc_semver::RustcVersion;
use rustc_span::symbol::sym;
use rustc_span::Span;
use std::borrow::Cow;

type McfResult = Result<(), (Span, Cow<'static, str>)>;

pub fn is_min_const_fn<'tcx>(tcx: TyCtxt<'tcx>, body: &Body<'tcx>, msrv: &Msrv) -> McfResult {
    let def_id = body.source.def_id();
    let mut current = def_id;
    loop {
        let predicates = tcx.predicates_of(current);
        for (predicate, _) in predicates.predicates {
            match predicate.kind().skip_binder() {
                ty::PredicateKind::Clause(
                    ty::Clause::RegionOutlives(_)
                    | ty::Clause::TypeOutlives(_)
                    | ty::Clause::Projection(_)
                    | ty::Clause::Trait(..)
                    | ty::Clause::ConstArgHasType(..),
                )
                | ty::PredicateKind::WellFormed(_)
                | ty::PredicateKind::ConstEvaluatable(..)
                | ty::PredicateKind::ConstEquate(..)
                | ty::PredicateKind::TypeWellFormedFromEnv(..) => continue,
                ty::PredicateKind::AliasEq(..) => panic!("alias eq predicate on function: {predicate:#?}"),
                ty::PredicateKind::ObjectSafe(_) => panic!("object safe predicate on function: {predicate:#?}"),
                ty::PredicateKind::ClosureKind(..) => panic!("closure kind predicate on function: {predicate:#?}"),
                ty::PredicateKind::Subtype(_) => panic!("subtype predicate on function: {predicate:#?}"),
                ty::PredicateKind::Coerce(_) => panic!("coerce predicate on function: {predicate:#?}"),
                ty::PredicateKind::Ambiguous => panic!("ambiguous predicate on function: {predicate:#?}"),
            }
        }
        match predicates.parent {
            Some(parent) => current = parent,
            None => break,
        }
    }

    for local in &body.local_decls {
        check_ty(tcx, local.ty, local.source_info.span)?;
    }
    // impl trait is gone in MIR, so check the return type manually
    check_ty(
        tcx,
        tcx.fn_sig(def_id).subst_identity().output().skip_binder(),
        body.local_decls.iter().next().unwrap().source_info.span,
    )?;

    for bb in body.basic_blocks.iter() {
        check_terminator(tcx, body, bb.terminator(), msrv)?;
        for stmt in &bb.statements {
            check_statement(tcx, body, def_id, stmt)?;
        }
    }
    Ok(())
}

fn check_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, span: Span) -> McfResult {
    for arg in ty.walk() {
        let ty = match arg.unpack() {
            GenericArgKind::Type(ty) => ty,

            // No constraints on lifetimes or constants, except potentially
            // constants' types, but `walk` will get to them as well.
            GenericArgKind::Lifetime(_) | GenericArgKind::Const(_) => continue,
        };

        match ty.kind() {
            ty::Ref(_, _, hir::Mutability::Mut) => {
                return Err((span, "mutable references in const fn are unstable".into()));
            },
            ty::Alias(ty::Opaque, ..) => return Err((span, "`impl Trait` in const fn is unstable".into())),
            ty::FnPtr(..) => {
                return Err((span, "function pointers in const fn are unstable".into()));
            },
            ty::Dynamic(preds, _, _) => {
                for pred in preds.iter() {
                    match pred.skip_binder() {
                        ty::ExistentialPredicate::AutoTrait(_) | ty::ExistentialPredicate::Projection(_) => {
                            return Err((
                                span,
                                "trait bounds other than `Sized` \
                                 on const fn parameters are unstable"
                                    .into(),
                            ));
                        },
                        ty::ExistentialPredicate::Trait(trait_ref) => {
                            if Some(trait_ref.def_id) != tcx.lang_items().sized_trait() {
                                return Err((
                                    span,
                                    "trait bounds other than `Sized` \
                                     on const fn parameters are unstable"
                                        .into(),
                                ));
                            }
                        },
                    }
                }
            },
            _ => {},
        }
    }
    Ok(())
}

fn check_rvalue<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    rvalue: &Rvalue<'tcx>,
    span: Span,
) -> McfResult {
    match rvalue {
        Rvalue::ThreadLocalRef(_) => Err((span, "cannot access thread local storage in const fn".into())),
        Rvalue::Len(place) | Rvalue::Discriminant(place) | Rvalue::Ref(_, _, place) | Rvalue::AddressOf(_, place) => {
            check_place(tcx, *place, span, body)
        },
        Rvalue::CopyForDeref(place) => check_place(tcx, *place, span, body),
        Rvalue::Repeat(operand, _)
        | Rvalue::Use(operand)
        | Rvalue::Cast(
            CastKind::PointerFromExposedAddress
            | CastKind::IntToInt
            | CastKind::FloatToInt
            | CastKind::IntToFloat
            | CastKind::FloatToFloat
            | CastKind::FnPtrToPtr
            | CastKind::PtrToPtr
            | CastKind::Pointer(PointerCast::MutToConstPointer | PointerCast::ArrayToPointer),
            operand,
            _,
        ) => check_operand(tcx, operand, span, body),
        Rvalue::Cast(
            CastKind::Pointer(
                PointerCast::UnsafeFnPointer | PointerCast::ClosureFnPointer(_) | PointerCast::ReifyFnPointer,
            ),
            _,
            _,
        ) => Err((span, "function pointer casts are not allowed in const fn".into())),
        Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), op, cast_ty) => {
            let pointee_ty = if let Some(deref_ty) = cast_ty.builtin_deref(true) {
                deref_ty.ty
            } else {
                // We cannot allow this for now.
                return Err((span, "unsizing casts are only allowed for references right now".into()));
            };
            let unsized_ty = tcx.struct_tail_erasing_lifetimes(pointee_ty, tcx.param_env(def_id));
            if let ty::Slice(_) | ty::Str = unsized_ty.kind() {
                check_operand(tcx, op, span, body)?;
                // Casting/coercing things to slices is fine.
                Ok(())
            } else {
                // We just can't allow trait objects until we have figured out trait method calls.
                Err((span, "unsizing casts are not allowed in const fn".into()))
            }
        },
        Rvalue::Cast(CastKind::PointerExposeAddress, _, _) => {
            Err((span, "casting pointers to ints is unstable in const fn".into()))
        },
        Rvalue::Cast(CastKind::DynStar, _, _) => {
            // FIXME(dyn-star)
            unimplemented!()
        },
        // binops are fine on integers
        Rvalue::BinaryOp(_, box (lhs, rhs)) | Rvalue::CheckedBinaryOp(_, box (lhs, rhs)) => {
            check_operand(tcx, lhs, span, body)?;
            check_operand(tcx, rhs, span, body)?;
            let ty = lhs.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() || ty.is_char() {
                Ok(())
            } else {
                Err((
                    span,
                    "only int, `bool` and `char` operations are stable in const fn".into(),
                ))
            }
        },
        Rvalue::NullaryOp(NullOp::SizeOf | NullOp::AlignOf, _) | Rvalue::ShallowInitBox(_, _) => Ok(()),
        Rvalue::UnaryOp(_, operand) => {
            let ty = operand.ty(body, tcx);
            if ty.is_integral() || ty.is_bool() {
                check_operand(tcx, operand, span, body)
            } else {
                Err((span, "only int and `bool` operations are stable in const fn".into()))
            }
        },
        Rvalue::Aggregate(_, operands) => {
            for operand in operands {
                check_operand(tcx, operand, span, body)?;
            }
            Ok(())
        },
    }
}

fn check_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    def_id: DefId,
    statement: &Statement<'tcx>,
) -> McfResult {
    let span = statement.source_info.span;
    match &statement.kind {
        StatementKind::Assign(box (place, rval)) => {
            check_place(tcx, *place, span, body)?;
            check_rvalue(tcx, body, def_id, rval, span)
        },

        StatementKind::FakeRead(box (_, place)) => check_place(tcx, *place, span, body),
        // just an assignment
        StatementKind::SetDiscriminant { place, .. } | StatementKind::Deinit(place) => {
            check_place(tcx, **place, span, body)
        },

        StatementKind::Intrinsic(box NonDivergingIntrinsic::Assume(op)) => check_operand(tcx, op, span, body),

        StatementKind::Intrinsic(box NonDivergingIntrinsic::CopyNonOverlapping(
            rustc_middle::mir::CopyNonOverlapping { dst, src, count },
        )) => {
            check_operand(tcx, dst, span, body)?;
            check_operand(tcx, src, span, body)?;
            check_operand(tcx, count, span, body)
        },
        // These are all NOPs
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Retag { .. }
        | StatementKind::AscribeUserType(..)
        | StatementKind::Coverage(..)
        | StatementKind::ConstEvalCounter
        | StatementKind::Nop => Ok(()),
    }
}

fn check_operand<'tcx>(tcx: TyCtxt<'tcx>, operand: &Operand<'tcx>, span: Span, body: &Body<'tcx>) -> McfResult {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => check_place(tcx, *place, span, body),
        Operand::Constant(c) => match c.check_static_ptr(tcx) {
            Some(_) => Err((span, "cannot access `static` items in const fn".into())),
            None => Ok(()),
        },
    }
}

fn check_place<'tcx>(tcx: TyCtxt<'tcx>, place: Place<'tcx>, span: Span, body: &Body<'tcx>) -> McfResult {
    let mut cursor = place.projection.as_ref();
    while let [ref proj_base @ .., elem] = *cursor {
        cursor = proj_base;
        match elem {
            ProjectionElem::Field(..) => {
                let base_ty = Place::ty_from(place.local, proj_base, body, tcx).ty;
                if let Some(def) = base_ty.ty_adt_def() {
                    // No union field accesses in `const fn`
                    if def.is_union() {
                        return Err((span, "accessing union fields is unstable".into()));
                    }
                }
            },
            ProjectionElem::ConstantIndex { .. }
            | ProjectionElem::OpaqueCast(..)
            | ProjectionElem::Downcast(..)
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Deref
            | ProjectionElem::Index(_) => {},
        }
    }

    Ok(())
}

fn check_terminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &Body<'tcx>,
    terminator: &Terminator<'tcx>,
    msrv: &Msrv,
) -> McfResult {
    let span = terminator.source_info.span;
    match &terminator.kind {
        TerminatorKind::FalseEdge { .. }
        | TerminatorKind::FalseUnwind { .. }
        | TerminatorKind::Goto { .. }
        | TerminatorKind::Return
        | TerminatorKind::Resume
        | TerminatorKind::Unreachable => Ok(()),

        TerminatorKind::Drop { place, .. } => check_place(tcx, *place, span, body),
        TerminatorKind::DropAndReplace { place, value, .. } => {
            check_place(tcx, *place, span, body)?;
            check_operand(tcx, value, span, body)
        },

        TerminatorKind::SwitchInt { discr, targets: _ } => check_operand(tcx, discr, span, body),

        TerminatorKind::Abort => Err((span, "abort is not stable in const fn".into())),
        TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => {
            Err((span, "const fn generators are unstable".into()))
        },

        TerminatorKind::Call {
            func,
            args,
            from_hir_call: _,
            destination: _,
            target: _,
            cleanup: _,
            fn_span: _,
        } => {
            let fn_ty = func.ty(body, tcx);
            if let ty::FnDef(fn_def_id, _) = *fn_ty.kind() {
                if !is_const_fn(tcx, fn_def_id, msrv) {
                    return Err((
                        span,
                        format!(
                            "can only call other `const fn` within a `const fn`, \
                             but `{func:?}` is not stable as `const fn`",
                        )
                        .into(),
                    ));
                }

                // HACK: This is to "unstabilize" the `transmute` intrinsic
                // within const fns. `transmute` is allowed in all other const contexts.
                // This won't really scale to more intrinsics or functions. Let's allow const
                // transmutes in const fn before we add more hacks to this.
                if tcx.is_intrinsic(fn_def_id) && tcx.item_name(fn_def_id) == sym::transmute {
                    return Err((
                        span,
                        "can only call `transmute` from const items, not `const fn`".into(),
                    ));
                }

                check_operand(tcx, func, span, body)?;

                for arg in args {
                    check_operand(tcx, arg, span, body)?;
                }
                Ok(())
            } else {
                Err((span, "can only call other const fns within const fn".into()))
            }
        },

        TerminatorKind::Assert {
            cond,
            expected: _,
            msg: _,
            target: _,
            cleanup: _,
        } => check_operand(tcx, cond, span, body),

        TerminatorKind::InlineAsm { .. } => Err((span, "cannot use inline assembly in const fn".into())),
    }
}

fn is_const_fn(tcx: TyCtxt<'_>, def_id: DefId, msrv: &Msrv) -> bool {
    tcx.is_const_fn(def_id)
        && tcx.lookup_const_stability(def_id).map_or(true, |const_stab| {
            if let rustc_attr::StabilityLevel::Stable { since, .. } = const_stab.level {
                // Checking MSRV is manually necessary because `rustc` has no such concept. This entire
                // function could be removed if `rustc` provided a MSRV-aware version of `is_const_fn`.
                // as a part of an unimplemented MSRV check https://github.com/rust-lang/rust/issues/65262.

                // HACK(nilstrieb): CURRENT_RUSTC_VERSION can return versions like 1.66.0-dev. `rustc-semver`
                // doesn't accept                  the `-dev` version number so we have to strip it
                // off.
                let short_version = since
                    .as_str()
                    .split('-')
                    .next()
                    .expect("rustc_attr::StabilityLevel::Stable::since` is empty");

                let since = rustc_span::Symbol::intern(short_version);

                msrv.meets(RustcVersion::parse(since.as_str()).unwrap_or_else(|err| {
                    panic!("`rustc_attr::StabilityLevel::Stable::since` is ill-formatted: `{since}`, {err:?}")
                }))
            } else {
                // Unstable const fn with the feature enabled.
                msrv.current().is_none()
            }
        })
}
