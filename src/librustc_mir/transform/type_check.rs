// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pass type-checks the MIR to ensure it is not broken.
#![allow(unreachable_code)]

use rustc::infer::{self, InferCtxt, InferOk};
use rustc::traits::{self, Reveal};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, Ty, TyCtxt, TypeVariants};
use rustc::mir::*;
use rustc::mir::tcx::LvalueTy;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::mir::visit::Visitor;
use std::fmt;
use syntax::ast;
use syntax_pos::{Span, DUMMY_SP};

use rustc_data_structures::indexed_vec::Idx;

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $context.tcx().sess.span_warn(
            $context.last_span,
            &format!("broken MIR ({:?}): {}", $elem, format!($($message)*))
        )
    })
}

macro_rules! span_mirbug_and_err {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        {
            $context.tcx().sess.span_warn(
                $context.last_span,
                &format!("broken MIR ({:?}): {:?}", $elem, format!($($message)*))
            );
            $context.error()
        }
    })
}

enum FieldAccessError {
    OutOfRange { field_count: usize }
}

/// Verifies that MIR types are sane to not crash further checks.
///
/// The sanitize_XYZ methods here take an MIR object and compute its
/// type, calling `span_mirbug` and returning an error type if there
/// is a problem.
struct TypeVerifier<'a, 'b: 'a, 'gcx: 'b+'tcx, 'tcx: 'b> {
    cx: &'a mut TypeChecker<'b, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    last_span: Span,
    errors_reported: bool
}

impl<'a, 'b, 'gcx, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'gcx, 'tcx> {
    fn visit_span(&mut self, span: &Span) {
        if *span != DUMMY_SP {
            self.last_span = *span;
        }
    }

    fn visit_lvalue(&mut self,
                    lvalue: &Lvalue<'tcx>,
                    _context: visit::LvalueContext,
                    location: Location) {
        self.sanitize_lvalue(lvalue, location);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>, location: Location) {
        self.super_constant(constant, location);
        self.sanitize_type(constant, constant.ty);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        self.super_rvalue(rvalue, location);
        if let Some(ty) = rvalue.ty(self.mir, self.tcx()) {
            self.sanitize_type(rvalue, ty);
        }
    }

    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        self.sanitize_type(&"return type", mir.return_ty);
        for local_decl in &mir.local_decls {
            self.sanitize_type(local_decl, local_decl.ty);
        }
        if self.errors_reported {
            return;
        }
        self.super_mir(mir);
    }
}

impl<'a, 'b, 'gcx, 'tcx> TypeVerifier<'a, 'b, 'gcx, 'tcx> {
    fn new(cx: &'a mut TypeChecker<'b, 'gcx, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        TypeVerifier {
            cx: cx,
            mir: mir,
            last_span: mir.span,
            errors_reported: false
        }
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.cx.infcx.tcx
    }

    fn sanitize_type(&mut self, parent: &fmt::Debug, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.needs_infer() || ty.has_escaping_regions() || ty.references_error() {
            span_mirbug_and_err!(self, parent, "bad type {:?}", ty)
        } else {
            ty
        }
    }

    fn sanitize_lvalue(&mut self, lvalue: &Lvalue<'tcx>, location: Location) -> LvalueTy<'tcx> {
        debug!("sanitize_lvalue: {:?}", lvalue);
        match *lvalue {
            Lvalue::Local(index) => LvalueTy::Ty { ty: self.mir.local_decls[index].ty },
            Lvalue::Static(def_id) =>
                LvalueTy::Ty { ty: self.tcx().item_type(def_id) },
            Lvalue::Projection(ref proj) => {
                let base_ty = self.sanitize_lvalue(&proj.base, location);
                if let LvalueTy::Ty { ty } = base_ty {
                    if ty.references_error() {
                        assert!(self.errors_reported);
                        return LvalueTy::Ty { ty: self.tcx().types.err };
                    }
                }
                self.sanitize_projection(base_ty, &proj.elem, lvalue, location)
            }
        }
    }

    fn sanitize_projection(&mut self,
                           base: LvalueTy<'tcx>,
                           pi: &LvalueElem<'tcx>,
                           lvalue: &Lvalue<'tcx>,
                           location: Location)
                           -> LvalueTy<'tcx> {
        debug!("sanitize_projection: {:?} {:?} {:?}", base, pi, lvalue);
        let tcx = self.tcx();
        let base_ty = base.to_ty(tcx);
        let span = self.last_span;
        match *pi {
            ProjectionElem::Deref => {
                let deref_ty = base_ty.builtin_deref(true, ty::LvaluePreference::NoPreference);
                LvalueTy::Ty {
                    ty: deref_ty.map(|t| t.ty).unwrap_or_else(|| {
                        span_mirbug_and_err!(
                            self, lvalue, "deref of non-pointer {:?}", base_ty)
                    })
                }
            }
            ProjectionElem::Index(ref i) => {
                self.visit_operand(i, location);
                let index_ty = i.ty(self.mir, tcx);
                if index_ty != tcx.types.usize {
                    LvalueTy::Ty {
                        ty: span_mirbug_and_err!(self, i, "index by non-usize {:?}", i)
                    }
                } else {
                    LvalueTy::Ty {
                        ty: base_ty.builtin_index().unwrap_or_else(|| {
                            span_mirbug_and_err!(
                                self, lvalue, "index of non-array {:?}", base_ty)
                        })
                    }
                }
            }
            ProjectionElem::ConstantIndex { .. } => {
                // consider verifying in-bounds
                LvalueTy::Ty {
                    ty: base_ty.builtin_index().unwrap_or_else(|| {
                        span_mirbug_and_err!(
                            self, lvalue, "index of non-array {:?}", base_ty)
                    })
                }
            }
            ProjectionElem::Subslice { from, to } => {
                LvalueTy::Ty {
                    ty: match base_ty.sty {
                        ty::TyArray(inner, size) => {
                            let min_size = (from as usize) + (to as usize);
                            if let Some(rest_size) = size.checked_sub(min_size) {
                                tcx.mk_array(inner, rest_size)
                            } else {
                                span_mirbug_and_err!(
                                    self, lvalue, "taking too-small slice of {:?}", base_ty)
                            }
                        }
                        ty::TySlice(..) => base_ty,
                        _ => {
                            span_mirbug_and_err!(
                                self, lvalue, "slice of non-array {:?}", base_ty)
                        }
                    }
                }
            }
            ProjectionElem::Downcast(adt_def1, index) =>
                match base_ty.sty {
                    ty::TyAdt(adt_def, substs) if adt_def.is_enum() && adt_def == adt_def1 => {
                        if index >= adt_def.variants.len() {
                            LvalueTy::Ty {
                                ty: span_mirbug_and_err!(
                                    self,
                                    lvalue,
                                    "cast to variant #{:?} but enum only has {:?}",
                                    index,
                                    adt_def.variants.len())
                            }
                        } else {
                            LvalueTy::Downcast {
                                adt_def: adt_def,
                                substs: substs,
                                variant_index: index
                            }
                        }
                    }
                    _ => LvalueTy::Ty {
                        ty: span_mirbug_and_err!(
                            self, lvalue, "can't downcast {:?} as {:?}",
                            base_ty, adt_def1)
                    }
                },
            ProjectionElem::Field(field, fty) => {
                let fty = self.sanitize_type(lvalue, fty);
                match self.field_ty(lvalue, base, field) {
                    Ok(ty) => {
                        if let Err(terr) = self.cx.eq_types(span, ty, fty) {
                            span_mirbug!(
                                self, lvalue, "bad field access ({:?}: {:?}): {:?}",
                                ty, fty, terr);
                        }
                    }
                    Err(FieldAccessError::OutOfRange { field_count }) => {
                        span_mirbug!(
                            self, lvalue, "accessed field #{} but variant only has {}",
                            field.index(), field_count)
                    }
                }
                LvalueTy::Ty { ty: fty }
            }
        }
    }

    fn error(&mut self) -> Ty<'tcx> {
        self.errors_reported = true;
        self.tcx().types.err
    }

    fn field_ty(&mut self,
                parent: &fmt::Debug,
                base_ty: LvalueTy<'tcx>,
                field: Field)
                -> Result<Ty<'tcx>, FieldAccessError>
    {
        let tcx = self.tcx();

        let (variant, substs) = match base_ty {
            LvalueTy::Downcast { adt_def, substs, variant_index } => {
                (&adt_def.variants[variant_index], substs)
            }
            LvalueTy::Ty { ty } => match ty.sty {
                ty::TyAdt(adt_def, substs) if adt_def.is_univariant() => {
                        (&adt_def.variants[0], substs)
                    }
                ty::TyClosure(def_id, substs) => {
                    return match substs.upvar_tys(def_id, tcx).nth(field.index()) {
                        Some(ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: substs.upvar_tys(def_id, tcx).count()
                        })
                    }
                }
                ty::TyTuple(tys) => {
                    return match tys.get(field.index()) {
                        Some(&ty) => Ok(ty),
                        None => Err(FieldAccessError::OutOfRange {
                            field_count: tys.len()
                        })
                    }
                }
                _ => return Ok(span_mirbug_and_err!(
                    self, parent, "can't project out of {:?}", base_ty))
            }
        };

        if let Some(field) = variant.fields.get(field.index()) {
            Ok(self.cx.normalize(&field.ty(tcx, substs)))
        } else {
            Err(FieldAccessError::OutOfRange { field_count: variant.fields.len() })
        }
    }
}

pub struct TypeChecker<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    fulfillment_cx: traits::FulfillmentContext<'tcx>,
    last_span: Span,
    body_id: ast::NodeId,
}

impl<'a, 'gcx, 'tcx> TypeChecker<'a, 'gcx, 'tcx> {
    fn new(infcx: &'a InferCtxt<'a, 'gcx, 'tcx>, body_id: ast::NodeId) -> Self {
        TypeChecker {
            infcx: infcx,
            fulfillment_cx: traits::FulfillmentContext::new(),
            last_span: DUMMY_SP,
            body_id: body_id,
        }
    }

    fn misc(&self, span: Span) -> traits::ObligationCause<'tcx> {
        traits::ObligationCause::misc(span, self.body_id)
    }

    pub fn register_infer_ok_obligations<T>(&mut self, infer_ok: InferOk<'tcx, T>) -> T {
        for obligation in infer_ok.obligations {
            self.fulfillment_cx.register_predicate_obligation(self.infcx, obligation);
        }
        infer_ok.value
    }

    fn sub_types(&mut self, sup: Ty<'tcx>, sub: Ty<'tcx>)
                 -> infer::UnitResult<'tcx>
    {
        self.infcx.sub_types(false, &self.misc(self.last_span), sup, sub)
            .map(|ok| self.register_infer_ok_obligations(ok))
    }

    fn eq_types(&mut self, span: Span, a: Ty<'tcx>, b: Ty<'tcx>)
                -> infer::UnitResult<'tcx>
    {
        self.infcx.eq_types(false, &self.misc(span), a, b)
            .map(|ok| self.register_infer_ok_obligations(ok))
    }

    fn tcx(&self) -> TyCtxt<'a, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn check_stmt(&mut self, mir: &Mir<'tcx>, stmt: &Statement<'tcx>) {
        debug!("check_stmt: {:?}", stmt);
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(ref lv, ref rv) => {
                let lv_ty = lv.ty(mir, tcx).to_ty(tcx);
                let rv_ty = rv.ty(mir, tcx);
                if let Some(rv_ty) = rv_ty {
                    if let Err(terr) = self.sub_types(rv_ty, lv_ty) {
                        span_mirbug!(self, stmt, "bad assignment ({:?} = {:?}): {:?}",
                                     lv_ty, rv_ty, terr);
                    }
                // FIXME: rvalue with undeterminable type - e.g. inline
                // asm.
                }
            }
            StatementKind::SetDiscriminant{ ref lvalue, variant_index } => {
                let lvalue_type = lvalue.ty(mir, tcx).to_ty(tcx);
                let adt = match lvalue_type.sty {
                    TypeVariants::TyAdt(adt, _) if adt.is_enum() => adt,
                    _ => {
                        span_bug!(stmt.source_info.span,
                                  "bad set discriminant ({:?} = {:?}): lhs is not an enum",
                                  lvalue,
                                  variant_index);
                    }
                };
                if variant_index >= adt.variants.len() {
                     span_bug!(stmt.source_info.span,
                               "bad set discriminant ({:?} = {:?}): value of of range",
                               lvalue,
                               variant_index);
                };
            }
            StatementKind::StorageLive(ref lv) |
            StatementKind::StorageDead(ref lv) => {
                match *lv {
                    Lvalue::Local(_) => {}
                    _ => {
                        span_mirbug!(self, stmt, "bad lvalue: expected local");
                    }
                }
            }
            StatementKind::Nop => {}
        }
    }

    fn check_terminator(&mut self,
                        mir: &Mir<'tcx>,
                        term: &Terminator<'tcx>) {
        debug!("check_terminator: {:?}", term);
        let tcx = self.tcx();
        match term.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::Drop { .. } => {
                // no checks needed for these
            }


            TerminatorKind::DropAndReplace {
                ref location,
                ref value,
                ..
            } => {
                let lv_ty = location.ty(mir, tcx).to_ty(tcx);
                let rv_ty = value.ty(mir, tcx);
                if let Err(terr) = self.sub_types(rv_ty, lv_ty) {
                    span_mirbug!(self, term, "bad DropAndReplace ({:?} = {:?}): {:?}",
                                 lv_ty, rv_ty, terr);
                }
            }

            TerminatorKind::If { ref cond, .. } => {
                let cond_ty = cond.ty(mir, tcx);
                match cond_ty.sty {
                    ty::TyBool => {}
                    _ => {
                        span_mirbug!(self, term, "bad If ({:?}, not bool", cond_ty);
                    }
                }
            }
            TerminatorKind::SwitchInt { ref discr, switch_ty, .. } => {
                let discr_ty = discr.ty(mir, tcx).to_ty(tcx);
                if let Err(terr) = self.sub_types(discr_ty, switch_ty) {
                    span_mirbug!(self, term, "bad SwitchInt ({:?} on {:?}): {:?}",
                                 switch_ty, discr_ty, terr);
                }
                if !switch_ty.is_integral() && !switch_ty.is_char() &&
                    !switch_ty.is_bool()
                {
                    span_mirbug!(self, term, "bad SwitchInt discr ty {:?}",switch_ty);
                }
                // FIXME: check the values
            }
            TerminatorKind::Switch { ref discr, adt_def, ref targets } => {
                let discr_ty = discr.ty(mir, tcx).to_ty(tcx);
                match discr_ty.sty {
                    ty::TyAdt(def, _) if def.is_enum() &&
                                         def == adt_def &&
                                         adt_def.variants.len() == targets.len()
                      => {},
                    _ => {
                        span_mirbug!(self, term, "bad Switch ({:?} on {:?})",
                                     adt_def, discr_ty);
                    }
                }
            }
            TerminatorKind::Call { ref func, ref args, ref destination, .. } => {
                let func_ty = func.ty(mir, tcx);
                debug!("check_terminator: call, func_ty={:?}", func_ty);
                let func_ty = match func_ty.sty {
                    ty::TyFnDef(.., func_ty) | ty::TyFnPtr(func_ty) => func_ty,
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let sig = tcx.erase_late_bound_regions(&func_ty.sig);
                let sig = self.normalize(&sig);
                self.check_call_dest(mir, term, &sig, destination);

                if self.is_box_free(func) {
                    self.check_box_free_inputs(mir, term, &sig, args);
                } else {
                    self.check_call_inputs(mir, term, &sig, args);
                }
            }
            TerminatorKind::Assert { ref cond, ref msg, .. } => {
                let cond_ty = cond.ty(mir, tcx);
                if cond_ty != tcx.types.bool {
                    span_mirbug!(self, term, "bad Assert ({:?}, not bool", cond_ty);
                }

                if let AssertMessage::BoundsCheck { ref len, ref index } = *msg {
                    if len.ty(mir, tcx) != tcx.types.usize {
                        span_mirbug!(self, len, "bounds-check length non-usize {:?}", len)
                    }
                    if index.ty(mir, tcx) != tcx.types.usize {
                        span_mirbug!(self, index, "bounds-check index non-usize {:?}", index)
                    }
                }
            }
        }
    }

    fn check_call_dest(&mut self,
                       mir: &Mir<'tcx>,
                       term: &Terminator<'tcx>,
                       sig: &ty::FnSig<'tcx>,
                       destination: &Option<(Lvalue<'tcx>, BasicBlock)>) {
        let tcx = self.tcx();
        match *destination {
            Some((ref dest, _)) => {
                let dest_ty = dest.ty(mir, tcx).to_ty(tcx);
                if let Err(terr) = self.sub_types(sig.output(), dest_ty) {
                    span_mirbug!(self, term,
                                 "call dest mismatch ({:?} <- {:?}): {:?}",
                                 dest_ty, sig.output(), terr);
                }
            },
            None => {
                // FIXME(canndrew): This is_never should probably be an is_uninhabited
                if !sig.output().is_never() {
                    span_mirbug!(self, term, "call to converging function {:?} w/o dest", sig);
                }
            },
        }
    }

    fn check_call_inputs(&mut self,
                         mir: &Mir<'tcx>,
                         term: &Terminator<'tcx>,
                         sig: &ty::FnSig<'tcx>,
                         args: &[Operand<'tcx>])
    {
        debug!("check_call_inputs({:?}, {:?})", sig, args);
        if args.len() < sig.inputs().len() ||
           (args.len() > sig.inputs().len() && !sig.variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }
        for (n, (fn_arg, op_arg)) in sig.inputs().iter().zip(args).enumerate() {
            let op_arg_ty = op_arg.ty(mir, self.tcx());
            if let Err(terr) = self.sub_types(op_arg_ty, fn_arg) {
                span_mirbug!(self, term, "bad arg #{:?} ({:?} <- {:?}): {:?}",
                             n, fn_arg, op_arg_ty, terr);
            }
        }
    }

    fn is_box_free(&self, operand: &Operand<'tcx>) -> bool {
        match operand {
            &Operand::Constant(Constant {
                literal: Literal::Item { def_id, .. }, ..
            }) => {
                Some(def_id) == self.tcx().lang_items.box_free_fn()
            }
            _ => false,
        }
    }

    fn check_box_free_inputs(&mut self,
                             mir: &Mir<'tcx>,
                             term: &Terminator<'tcx>,
                             sig: &ty::FnSig<'tcx>,
                             args: &[Operand<'tcx>])
    {
        debug!("check_box_free_inputs");

        // box_free takes a Box as a pointer. Allow for that.

        if sig.inputs().len() != 1 {
            span_mirbug!(self, term, "box_free should take 1 argument");
            return;
        }

        let pointee_ty = match sig.inputs()[0].sty {
            ty::TyRawPtr(mt) => mt.ty,
            _ => {
                span_mirbug!(self, term, "box_free should take a raw ptr");
                return;
            }
        };

        if args.len() != 1 {
            span_mirbug!(self, term, "box_free called with wrong # of args");
            return;
        }

        let arg_ty = match args[0].ty(mir, self.tcx()).sty {
            ty::TyRawPtr(mt) => mt.ty,
            ty::TyBox(ty) => ty,
            _ => {
                span_mirbug!(self, term, "box_free called with bad arg ty");
                return;
            }
        };

        if let Err(terr) = self.sub_types(arg_ty, pointee_ty) {
            span_mirbug!(self, term, "bad box_free arg ({:?} <- {:?}): {:?}",
                         pointee_ty, arg_ty, terr);
        }
    }

    fn check_iscleanup(&mut self, mir: &Mir<'tcx>, block: &BasicBlockData<'tcx>)
    {
        let is_cleanup = block.is_cleanup;
        self.last_span = block.terminator().source_info.span;
        match block.terminator().kind {
            TerminatorKind::Goto { target } =>
                self.assert_iscleanup(mir, block, target, is_cleanup),
            TerminatorKind::If { targets: (on_true, on_false), .. } => {
                self.assert_iscleanup(mir, block, on_true, is_cleanup);
                self.assert_iscleanup(mir, block, on_false, is_cleanup);
            }
            TerminatorKind::Switch { ref targets, .. } |
            TerminatorKind::SwitchInt { ref targets, .. } => {
                for target in targets {
                    self.assert_iscleanup(mir, block, *target, is_cleanup);
                }
            }
            TerminatorKind::Resume => {
                if !is_cleanup {
                    span_mirbug!(self, block, "resume on non-cleanup block!")
                }
            }
            TerminatorKind::Return => {
                if is_cleanup {
                    span_mirbug!(self, block, "return on cleanup block")
                }
            }
            TerminatorKind::Unreachable => {}
            TerminatorKind::Drop { target, unwind, .. } |
            TerminatorKind::DropAndReplace { target, unwind, .. } |
            TerminatorKind::Assert { target, cleanup: unwind, .. } => {
                self.assert_iscleanup(mir, block, target, is_cleanup);
                if let Some(unwind) = unwind {
                    if is_cleanup {
                        span_mirbug!(self, block, "unwind on cleanup block")
                    }
                    self.assert_iscleanup(mir, block, unwind, true);
                }
            }
            TerminatorKind::Call { ref destination, cleanup, .. } => {
                if let &Some((_, target)) = destination {
                    self.assert_iscleanup(mir, block, target, is_cleanup);
                }
                if let Some(cleanup) = cleanup {
                    if is_cleanup {
                        span_mirbug!(self, block, "cleanup on cleanup block")
                    }
                    self.assert_iscleanup(mir, block, cleanup, true);
                }
            }
        }
    }

    fn assert_iscleanup(&mut self,
                        mir: &Mir<'tcx>,
                        ctxt: &fmt::Debug,
                        bb: BasicBlock,
                        iscleanuppad: bool)
    {
        if mir[bb].is_cleanup != iscleanuppad {
            span_mirbug!(self, ctxt, "cleanuppad mismatch: {:?} should be {:?}",
                         bb, iscleanuppad);
        }
    }

    fn typeck_mir(&mut self, mir: &Mir<'tcx>) {
        self.last_span = mir.span;
        debug!("run_on_mir: {:?}", mir.span);
        for block in mir.basic_blocks() {
            for stmt in &block.statements {
                if stmt.source_info.span != DUMMY_SP {
                    self.last_span = stmt.source_info.span;
                }
                self.check_stmt(mir, stmt);
            }

            self.check_terminator(mir, block.terminator());
            self.check_iscleanup(mir, block);
        }
    }


    fn normalize<T>(&mut self, value: &T) -> T
        where T: fmt::Debug + TypeFoldable<'tcx>
    {
        let mut selcx = traits::SelectionContext::new(self.infcx);
        let cause = traits::ObligationCause::misc(self.last_span, ast::CRATE_NODE_ID);
        let traits::Normalized { value, obligations } =
            traits::normalize(&mut selcx, cause, value);

        debug!("normalize: value={:?} obligations={:?}",
               value,
               obligations);

        let mut fulfill_cx = &mut self.fulfillment_cx;
        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(self.infcx, obligation);
        }

        value
    }

    fn verify_obligations(&mut self, mir: &Mir<'tcx>) {
        self.last_span = mir.span;
        if let Err(e) = self.fulfillment_cx.select_all_or_error(self.infcx) {
            span_mirbug!(self, "", "errors selecting obligation: {:?}",
                         e);
        }
    }
}

pub struct TypeckMir;

impl TypeckMir {
    pub fn new() -> Self {
        TypeckMir
    }
}

impl<'tcx> MirPass<'tcx> for TypeckMir {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>,
                    src: MirSource, mir: &mut Mir<'tcx>) {
        debug!("run_pass: {}", tcx.node_path_str(src.item_id()));

        if tcx.sess.err_count() > 0 {
            // compiling a broken program can obviously result in a
            // broken MIR, so try not to report duplicate errors.
            return;
        }
        let param_env = ty::ParameterEnvironment::for_item(tcx, src.item_id());
        tcx.infer_ctxt(param_env, Reveal::NotSpecializable).enter(|infcx| {
            let mut checker = TypeChecker::new(&infcx, src.item_id());
            {
                let mut verifier = TypeVerifier::new(&mut checker, mir);
                verifier.visit_mir(mir);
                if verifier.errors_reported {
                    // don't do further checks to avoid ICEs
                    return;
                }
            }
            checker.typeck_mir(mir);
            checker.verify_obligations(mir);
        });
    }
}

impl Pass for TypeckMir {
}
