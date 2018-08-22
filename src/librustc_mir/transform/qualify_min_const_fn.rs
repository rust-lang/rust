use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::mir::*;
use rustc::ty::{self, Predicate, TyCtxt};
use std::borrow::Cow;
use syntax_pos::Span;

mod helper {
    pub struct IsMinConstFn(());
    /// This should only ever be used *once* and then passed around as a token.
    pub fn ensure_that_you_really_intended_to_create_an_instance_of_this() -> IsMinConstFn {
        IsMinConstFn(())
    }
}

use self::helper::*;

type McfResult = Result<IsMinConstFn, (Span, Cow<'static, str>)>;

pub fn is_min_const_fn(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    def_id: DefId,
    mir: &'a Mir<'tcx>,
) -> McfResult {
    let mut current = def_id;
    loop {
        let predicates = tcx.predicates_of(current);
        for predicate in &predicates.predicates {
            match predicate {
                | Predicate::RegionOutlives(_)
                | Predicate::TypeOutlives(_)
                | Predicate::WellFormed(_)
                | Predicate::ConstEvaluatable(..) => continue,
                | Predicate::ObjectSafe(_) => {
                    bug!("object safe predicate on function: {:#?}", predicate)
                }
                Predicate::ClosureKind(..) => {
                    bug!("closure kind predicate on function: {:#?}", predicate)
                }
                Predicate::Subtype(_) => bug!("subtype predicate on function: {:#?}", predicate),
                Predicate::Projection(_) => {
                    let span = tcx.def_span(current);
                    // we'll hit a `Predicate::Trait` later which will report an error
                    tcx.sess
                        .delay_span_bug(span, "projection without trait bound");
                    continue;
                }
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

    let mut token = ensure_that_you_really_intended_to_create_an_instance_of_this();

    for local in mir.vars_iter() {
        return Err((
            mir.local_decls[local].source_info.span,
            "local variables in const fn are unstable".into(),
        ));
    }
    for local in &mir.local_decls {
        token = check_ty(tcx, local.ty, local.source_info.span, token)?;
    }
    // impl trait is gone in MIR, so check the return type manually
    token = check_ty(
        tcx,
        tcx.fn_sig(def_id).output().skip_binder(),
        mir.local_decls.iter().next().unwrap().source_info.span,
        token,
    )?;

    for bb in mir.basic_blocks() {
        token = check_terminator(tcx, mir, bb.terminator(), token)?;
        for stmt in &bb.statements {
            token = check_statement(tcx, mir, stmt, token)?;
        }
    }
    Ok(token)
}

fn check_ty(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    ty: ty::Ty<'tcx>,
    span: Span,
    token: IsMinConstFn,
) -> McfResult {
    for ty in ty.walk() {
        match ty.sty {
            ty::Ref(_, _, hir::Mutability::MutMutable) => return Err((
                span,
                "mutable references in const fn are unstable".into(),
            )),
            ty::Anon(..) => return Err((span, "`impl Trait` in const fn is unstable".into())),
            ty::FnPtr(..) => {
                return Err((span, "function pointers in const fn are unstable".into()))
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
    Ok(token)
}

fn check_rvalue(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    rvalue: &Rvalue<'tcx>,
    span: Span,
    token: IsMinConstFn,
) -> McfResult {
    match rvalue {
        Rvalue::Repeat(operand, _) | Rvalue::Use(operand) => {
            check_operand(tcx, mir, operand, span, token)
        }
        Rvalue::Len(place) | Rvalue::Discriminant(place) | Rvalue::Ref(_, _, place) => {
            check_place(tcx, mir, place, span, token, PlaceMode::Read)
        }
        Rvalue::Cast(_, operand, cast_ty) => {
            use rustc::ty::cast::CastTy;
            let cast_in = CastTy::from_ty(operand.ty(mir, tcx)).expect("bad input type for cast");
            let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
            match (cast_in, cast_out) {
                (CastTy::Ptr(_), CastTy::Int(_)) | (CastTy::FnPtr, CastTy::Int(_)) => Err((
                    span,
                    "casting pointers to ints is unstable in const fn".into(),
                )),
                (CastTy::RPtr(_), CastTy::Float) => bug!(),
                (CastTy::RPtr(_), CastTy::Int(_)) => bug!(),
                (CastTy::Ptr(_), CastTy::RPtr(_)) => bug!(),
                _ => check_operand(tcx, mir, operand, span, token),
            }
        }
        // binops are fine on integers
        Rvalue::BinaryOp(_, lhs, rhs) | Rvalue::CheckedBinaryOp(_, lhs, rhs) => {
            let token = check_operand(tcx, mir, lhs, span, token)?;
            let token = check_operand(tcx, mir, rhs, span, token)?;
            let ty = lhs.ty(mir, tcx);
            if ty.is_integral() || ty.is_bool() || ty.is_char() {
                Ok(token)
            } else {
                Err((
                    span,
                    "only int, `bool` and `char` operations are stable in const fn".into(),
                ))
            }
        }
        // checked by regular const fn checks
        Rvalue::NullaryOp(..) => Ok(token),
        Rvalue::UnaryOp(_, operand) => {
            let ty = operand.ty(mir, tcx);
            if ty.is_integral() || ty.is_bool() {
                check_operand(tcx, mir, operand, span, token)
            } else {
                Err((
                    span,
                    "only int and `bool` operations are stable in const fn".into(),
                ))
            }
        }
        Rvalue::Aggregate(_, operands) => {
            let mut token = token;
            for operand in operands {
                token = check_operand(tcx, mir, operand, span, token)?;
            }
            Ok(token)
        }
    }
}

enum PlaceMode {
    Assign,
    Read,
}

fn check_statement(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    statement: &Statement<'tcx>,
    token: IsMinConstFn,
) -> McfResult {
    let span = statement.source_info.span;
    match &statement.kind {
        StatementKind::Assign(place, rval) => {
            let token = check_place(tcx, mir, place, span, token, PlaceMode::Assign)?;
            check_rvalue(tcx, mir, rval, span, token)
        }

        StatementKind::ReadForMatch(_) => Err((span, "match in const fn is unstable".into())),

        // just an assignment
        StatementKind::SetDiscriminant { .. } => Ok(token),

        | StatementKind::InlineAsm { .. } => {
            Err((span, "cannot use inline assembly in const fn".into()))
        }

        // These are all NOPs
        | StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Validate(..)
        | StatementKind::EndRegion(_)
        | StatementKind::UserAssertTy(..)
        | StatementKind::Nop => Ok(token),
    }
}

fn check_operand(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    operand: &Operand<'tcx>,
    span: Span,
    token: IsMinConstFn,
) -> McfResult {
    match operand {
        Operand::Move(place) | Operand::Copy(place) => {
            check_place(tcx, mir, place, span, token, PlaceMode::Read)
        }
        Operand::Constant(_) => Ok(token),
    }
}

fn check_place(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    place: &Place<'tcx>,
    span: Span,
    token: IsMinConstFn,
    mode: PlaceMode,
) -> McfResult {
    match place {
        Place::Local(l) => match mode {
            PlaceMode::Assign => match mir.local_kind(*l) {
                LocalKind::Temp | LocalKind::ReturnPointer => Ok(token),
                LocalKind::Arg | LocalKind::Var => {
                    Err((span, "assignments in const fn are unstable".into()))
                }
            },
            PlaceMode::Read => Ok(token),
        },
        // promoteds are always fine, they are essentially constants
        Place::Promoted(_) => Ok(token),
        Place::Static(_) => Err((span, "cannot access `static` items in const fn".into())),
        Place::Projection(proj) => {
            match proj.elem {
                | ProjectionElem::Deref | ProjectionElem::Field(..) | ProjectionElem::Index(_) => {
                    check_place(tcx, mir, &proj.base, span, token, mode)
                }
                // slice patterns are unstable
                | ProjectionElem::ConstantIndex { .. } | ProjectionElem::Subslice { .. } => {
                    return Err((span, "slice patterns in const fn are unstable".into()))
                }
                | ProjectionElem::Downcast(..) => {
                    Err((span, "`match` or `if let` in `const fn` is unstable".into()))
                }
            }
        }
    }
}

fn check_terminator(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    terminator: &Terminator<'tcx>,
    token: IsMinConstFn,
) -> McfResult {
    let span = terminator.source_info.span;
    match &terminator.kind {
        | TerminatorKind::Goto { .. }
        | TerminatorKind::Return
        | TerminatorKind::Resume => Ok(token),

        TerminatorKind::Drop { location, .. } => {
            check_place(tcx, mir, location, span, token, PlaceMode::Read)
        }
        TerminatorKind::DropAndReplace { location, value, .. } => {
            let token = check_place(tcx, mir, location, span, token, PlaceMode::Read)?;
            check_operand(tcx, mir, value, span, token)
        },
        TerminatorKind::SwitchInt { .. } => Err((
            span,
            "`if`, `match`, `&&` and `||` are not stable in const fn".into(),
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
            destination: _,
            cleanup: _,
        } => {
            let fn_ty = func.ty(mir, tcx);
            if let ty::FnDef(def_id, _) = fn_ty.sty {
                if tcx.is_min_const_fn(def_id) {
                    let mut token = check_operand(tcx, mir, func, span, token)?;

                    for arg in args {
                        token = check_operand(tcx, mir, arg, span, token)?;
                    }
                    Ok(token)
                } else {
                    Err((
                        span,
                        "can only call other `min_const_fn` within a `min_const_fn`".into(),
                    ))
                }
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
        } => check_operand(tcx, mir, cond, span, token),

        | TerminatorKind::FalseEdges { .. } | TerminatorKind::FalseUnwind { .. } => span_bug!(
            terminator.source_info.span,
            "min_const_fn encountered `{:#?}`",
            terminator
        ),
    }
}
