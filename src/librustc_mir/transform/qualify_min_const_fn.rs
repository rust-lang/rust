use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::mir::*;
use rustc::mir::visit::{Visitor, TyContext, PlaceContext};
use rustc::ty::{self, Predicate, Ty, TyCtxt, adjustment::{PointerCast}};
use rustc_target::spec::abi;
use std::borrow::Cow;
use syntax_pos::Span;
use syntax::symbol::{sym, Symbol};
use syntax::attr;

type McfResult = Result<(), (Span, Cow<'static, str>)>;

pub fn is_min_const_fn(tcx: TyCtxt<'tcx>, def_id: DefId, body: &'a Body<'tcx>) -> McfResult {
    let mut current = def_id;
    loop {
        let predicates = tcx.predicates_of(current);
        for (predicate, _) in predicates.predicates {
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
                    match pred.skip_binder().self_ty().kind {
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

    let mut visitor = IsMinConstFn {
        tcx,
        body,
        def_id,
        span: body.span,
    };

    // We could do most of this simply by calling `visit_body`. However, we change the order since
    // we want to check each parameter **before** the function body, and the terminator **before**
    // the statements.
    for (local, decl) in body.local_decls.iter_enumerated() {
        visitor.visit_local_decl(local, decl)?;
    }

    // impl trait is gone in MIR, so check the return type manually
    visitor.span = body.local_decls[RETURN_PLACE].source_info.span;
    let ret_ty_ctx = TyContext::ReturnTy(body.local_decls.iter().next().unwrap().source_info);
    visitor.visit_ty(tcx.fn_sig(def_id).output().skip_binder(), ret_ty_ctx)?;

    for (bb, block) in body.basic_blocks().iter_enumerated() {
        let mut loc = body.terminator_loc(bb);
        visitor.visit_terminator(block.terminator(), loc)?;
        for (idx, stmt) in block.statements.iter().enumerate() {
            loc.statement_index = idx;
            visitor.visit_statement(stmt, loc)?;
        }
    }

    Ok(())
}

struct IsMinConstFn<'mir, 'tcx> {
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    span: Span,
    body: &'mir Body<'tcx>,
}

impl Visitor<'tcx, McfResult> for IsMinConstFn<'mir, 'tcx> {
    fn visit_source_info(&mut self, info: &SourceInfo) -> McfResult {
        self.span = info.span;
        Ok(())
    }

    fn visit_ty(&mut self, ty: Ty<'tcx>, _: TyContext) -> McfResult {
        let IsMinConstFn { tcx, def_id: fn_def_id, span, .. } = *self;
        for ty in ty.walk() {
            match ty.kind {
                ty::Ref(_, _, hir::Mutability::Mutable) => return Err((
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

        self.super_ty(ty)
    }

    fn visit_rvalue(
        &mut self,
        rvalue: &Rvalue<'tcx>,
        location: Location,
    ) -> McfResult {
        let IsMinConstFn { tcx, span, body, .. } = *self;
        match rvalue {
            Rvalue::Cast(CastKind::Misc, operand, cast_ty) => {
                use rustc::ty::cast::CastTy;
                let cast_in = CastTy::from_ty(operand.ty(body, tcx))
                    .expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty)
                    .expect("bad output type for cast");
                match (cast_in, cast_out) {
                    | (CastTy::Ptr(_), CastTy::Int(_))
                    | (CastTy::FnPtr, CastTy::Int(_))
                    => return Err((
                        span,
                        "casting pointers to ints is unstable in const fn".into(),
                    )),
                    (CastTy::RPtr(_), CastTy::Float) => bug!(),
                    (CastTy::RPtr(_), CastTy::Int(_)) => bug!(),
                    (CastTy::Ptr(_), CastTy::RPtr(_)) => bug!(),
                    _ => {}
                }
            }
            | Rvalue::Cast(CastKind::Pointer(PointerCast::UnsafeFnPointer), _, _)
            | Rvalue::Cast(CastKind::Pointer(PointerCast::ClosureFnPointer(_)), _, _)
            | Rvalue::Cast(CastKind::Pointer(PointerCast::ReifyFnPointer), _, _)
            => return Err((
                span,
                "function pointer casts are not allowed in const fn".into(),
            )),
            Rvalue::Cast(CastKind::Pointer(PointerCast::Unsize), _, _) => return Err((
                span,
                "unsizing casts are not allowed in const fn".into(),
            )),
            // binops are fine on integers
            Rvalue::BinaryOp(_, lhs, _) | Rvalue::CheckedBinaryOp(_, lhs, _) => {
                // FIXME: do we need to type check `rhs`?
                let ty = lhs.ty(body, tcx);
                if !(ty.is_integral() || ty.is_bool() || ty.is_char()) {
                    return Err((
                        span,
                        "only int, `bool` and `char` operations are stable in const fn".into(),
                    ))
                }
            }
            Rvalue::NullaryOp(NullOp::Box, _) => return Err((
                span,
                "heap allocations are not allowed in const fn".into(),
            )),
            Rvalue::UnaryOp(_, operand) => {
                let ty = operand.ty(body, tcx);
                if !(ty.is_integral() || ty.is_bool()) {
                    return Err((
                        span,
                        "only int and `bool` operations are stable in const fn".into(),
                    ))
                }
            }

            _ => {}
        }

        self.super_rvalue(rvalue, location)
    }

    fn visit_statement(
        &mut self,
        statement: &Statement<'tcx>,
        location: Location,
    ) -> McfResult {
        // Setting `span` here is necessary because we don't call `super_statement` until later.
        self.span = statement.source_info.span;
        let IsMinConstFn { tcx, span, .. } = *self;

        match &statement.kind {
            StatementKind::Assign(_) | StatementKind::SetDiscriminant { .. } => {
                self.super_statement(statement, location)
            }

            | StatementKind::FakeRead(FakeReadCause::ForMatchedPlace, _)
            if !tcx.features().const_if_match
            => {
                Err((span, "loops and conditional expressions are not stable in const fn".into()))
            }

            StatementKind::FakeRead(_, _) => {
                self.super_statement(statement, location)
            }

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

    fn visit_operand(&mut self, op: &Operand<'tcx>, location: Location) -> McfResult {
        let IsMinConstFn { tcx, span, .. } = *self;
        if let Operand::Constant(c) = op {
            if c.check_static_ptr(tcx).is_some() {
                return Err((span, "cannot access `static` items in const fn".into()));
            }
        }

        self.super_operand(op, location)
    }

    fn visit_projection_elem(
        &mut self,
        base: &PlaceBase<'tcx>,
        proj_base: &[PlaceElem<'tcx>],
        elem: &PlaceElem<'tcx>,
        context: PlaceContext,
        location: Location,
    ) -> McfResult {
        let IsMinConstFn { tcx, def_id, span, body } = *self;

        match elem {
            ProjectionElem::Downcast(..) if !tcx.features().const_if_match
                => return Err((span, "`match` or `if let` in `const fn` is unstable".into())),
            ProjectionElem::Downcast(_symbol, _variant_index) => {}

            ProjectionElem::Field(..) => {
                let base_ty = Place::ty_from(base, &proj_base, body, tcx).ty;
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
            | ProjectionElem::Subslice { .. }
            | ProjectionElem::Deref
            | ProjectionElem::Index(_) => {}
        }

        self.super_projection_elem(base, proj_base, elem, context, location)
    }

    fn visit_place_base(
        &mut self,
        base: &PlaceBase<'tcx>,
        _context: PlaceContext,
        _location: Location,
    ) -> McfResult {
        let IsMinConstFn { span, .. } = *self;

        // FIXME: should this call `check_ty` for each place?
        // self.super_place_base(base, context, location)?;

        match base {
            PlaceBase::Static(box Static { kind: StaticKind::Static, .. }) => {
                Err((span, "cannot access `static` items in const fn".into()))
            }
            PlaceBase::Local(_)
            | PlaceBase::Static(box Static { kind: StaticKind::Promoted(_, _), .. }) => Ok(()),
        }
    }

    fn visit_terminator_kind(
        &mut self,
        kind: &TerminatorKind<'tcx>,
        location: Location,
    ) -> McfResult {
        let IsMinConstFn { tcx, span, body, .. } = *self;

        match kind {
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::SwitchInt { .. }
            if !tcx.features().const_if_match
            => Err((
                span,
                "loops and conditional expressions are not stable in const fn".into(),
            )),

            | TerminatorKind::Goto { .. }
            | TerminatorKind::Return
            | TerminatorKind::Resume
            | TerminatorKind::Drop { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::FalseEdges { .. }
            | TerminatorKind::SwitchInt { .. }
            => self.super_terminator_kind(kind, location),

            // FIXME(ecstaticmorse): We probably want to allow `Unreachable` unconditionally.
            TerminatorKind::Unreachable if tcx.features().const_if_match => Ok(()),

            | TerminatorKind::Abort | TerminatorKind::Unreachable => {
                Err((span, "const fn with unreachable code is not stable".into()))
            }
            | TerminatorKind::GeneratorDrop | TerminatorKind::Yield { .. } => {
                Err((span, "const fn generators are unstable".into()))
            }

            TerminatorKind::Call {
                func,
                args: _,
                from_hir_call: _,
                destination: _,
                cleanup: _,
            } => {
                let fn_ty = func.ty(body, tcx);
                if let ty::FnDef(def_id, _) = fn_ty.kind {

                    // some intrinsics are waved hrough if called inside the
                    // standard library. Users never need to call them directly
                    match tcx.fn_sig(def_id).abi() {
                        abi::Abi::RustIntrinsic => if !is_intrinsic_whitelisted(tcx, def_id) {
                            return Err((
                                span,
                                "can only call a curated list of intrinsics \
                                 in `min_const_fn`".into(),
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

                    self.super_terminator_kind(kind, location)
                } else {
                    Err((span, "can only call other const fns within const fn".into()))
                }
            }

            TerminatorKind::FalseUnwind { .. } => {
                Err((span, "loops are not allowed in const fn".into()))
            },
        }
    }
}

/// Returns whether `allow_internal_unstable(..., <feature_gate>, ...)` is present.
fn feature_allowed(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    feature_gate: Symbol,
) -> bool {
    attr::allow_internal_unstable(&tcx.get_attrs(def_id), &tcx.sess.diagnostic())
        .map_or(false, |mut features| features.any(|name| name == feature_gate))
}

/// Returns `true` if the `def_id` refers to an intrisic which we've whitelisted
/// for being called from stable `const fn`s (`min_const_fn`).
///
/// Adding more intrinsics requires sign-off from @rust-lang/lang.
fn is_intrinsic_whitelisted(tcx: TyCtxt<'tcx>, def_id: DefId) -> bool {
    match &*tcx.item_name(def_id).as_str() {
        | "size_of"
        | "min_align_of"
        | "needs_drop"
        // Arithmetic:
        | "add_with_overflow" // ~> .overflowing_add
        | "sub_with_overflow" // ~> .overflowing_sub
        | "mul_with_overflow" // ~> .overflowing_mul
        | "wrapping_add" // ~> .wrapping_add
        | "wrapping_sub" // ~> .wrapping_sub
        | "wrapping_mul" // ~> .wrapping_mul
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
