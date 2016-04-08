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

use rustc::dep_graph::DepNode;
use rustc::infer::{self, InferCtxt, InferOk};
use rustc::traits::{self, ProjectionMode};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::{self, Ty, TyCtxt};
use rustc::mir::repr::*;
use rustc::mir::tcx::LvalueTy;
use rustc::mir::transform::{MirPass, Pass};
use rustc::mir::visit::{self, Visitor};
use std::fmt;
use syntax::ast::NodeId;
use syntax::codemap::{Span, DUMMY_SP};

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
struct TypeVerifier<'a, 'b: 'a, 'tcx: 'b> {
    cx: &'a mut TypeChecker<'b, 'tcx>,
    mir: &'a Mir<'tcx>,
    last_span: Span,
    errors_reported: bool
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'b, 'tcx> {
    fn visit_span(&mut self, span: &Span) {
        if *span != DUMMY_SP {
            self.last_span = *span;
        }
    }

    fn visit_lvalue(&mut self, lvalue: &Lvalue<'tcx>, _context: visit::LvalueContext) {
        self.sanitize_lvalue(lvalue);
    }

    fn visit_constant(&mut self, constant: &Constant<'tcx>) {
        self.super_constant(constant);
        self.sanitize_type(constant, constant.ty);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>) {
        self.super_rvalue(rvalue);
        if let Some(ty) = self.mir.rvalue_ty(self.tcx(), rvalue) {
            self.sanitize_type(rvalue, ty);
        }
    }

    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        if let ty::FnConverging(t) = mir.return_ty {
            self.sanitize_type(&"return type", t);
        }
        for var_decl in &mir.var_decls {
            self.sanitize_type(var_decl, var_decl.ty);
        }
        for (n, arg_decl) in mir.arg_decls.iter().enumerate() {
            self.sanitize_type(&(n, arg_decl), arg_decl.ty);
        }
        for (n, tmp_decl) in mir.temp_decls.iter().enumerate() {
            self.sanitize_type(&(n, tmp_decl), tmp_decl.ty);
        }
        if self.errors_reported {
            return;
        }
        self.super_mir(mir);
    }
}

impl<'a, 'b, 'tcx> TypeVerifier<'a, 'b, 'tcx> {
    fn new(cx: &'a mut TypeChecker<'b, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        TypeVerifier {
            cx: cx,
            mir: mir,
            last_span: mir.span,
            errors_reported: false
        }
    }

    fn tcx(&self) -> &'a TyCtxt<'tcx> {
        self.cx.infcx.tcx
    }

    fn infcx(&self) -> &'a InferCtxt<'a, 'tcx> {
        self.cx.infcx
    }

    fn sanitize_type(&mut self, parent: &fmt::Debug, ty: Ty<'tcx>) -> Ty<'tcx> {
        if ty.needs_infer() || ty.has_escaping_regions() || ty.references_error() {
            span_mirbug_and_err!(self, parent, "bad type {:?}", ty)
        } else {
            ty
        }
    }

    fn sanitize_lvalue(&mut self, lvalue: &Lvalue<'tcx>) -> LvalueTy<'tcx> {
        debug!("sanitize_lvalue: {:?}", lvalue);
        match *lvalue {
            Lvalue::Var(index) => LvalueTy::Ty { ty: self.mir.var_decls[index as usize].ty },
            Lvalue::Temp(index) =>
                LvalueTy::Ty { ty: self.mir.temp_decls[index as usize].ty },
            Lvalue::Arg(index) =>
                LvalueTy::Ty { ty: self.mir.arg_decls[index as usize].ty },
            Lvalue::Static(def_id) =>
                LvalueTy::Ty { ty: self.tcx().lookup_item_type(def_id).ty },
            Lvalue::ReturnPointer => {
                if let ty::FnConverging(return_ty) = self.mir.return_ty {
                    LvalueTy::Ty { ty: return_ty }
                } else {
                    LvalueTy::Ty {
                        ty: span_mirbug_and_err!(
                            self, lvalue, "return in diverging function")
                    }
                }
            }
            Lvalue::Projection(ref proj) => {
                let base_ty = self.sanitize_lvalue(&proj.base);
                if let LvalueTy::Ty { ty } = base_ty {
                    if ty.references_error() {
                        assert!(self.errors_reported);
                        return LvalueTy::Ty { ty: self.tcx().types.err };
                    }
                }
                self.sanitize_projection(base_ty, &proj.elem, lvalue)
            }
        }
    }

    fn sanitize_projection(&mut self,
                           base: LvalueTy<'tcx>,
                           pi: &LvalueElem<'tcx>,
                           lvalue: &Lvalue<'tcx>)
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
                self.visit_operand(i);
                let index_ty = self.mir.operand_ty(tcx, i);
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
            ProjectionElem::Downcast(adt_def1, index) =>
                match base_ty.sty {
                    ty::TyEnum(adt_def, substs) if adt_def == adt_def1 => {
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
                        if let Err(terr) = self.cx.mk_eqty(span, ty, fty) {
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
                ty::TyStruct(adt_def, substs) | ty::TyEnum(adt_def, substs)
                    if adt_def.is_univariant() => {
                        (&adt_def.variants[0], substs)
                    }
                ty::TyTuple(ref tys) | ty::TyClosure(_, box ty::ClosureSubsts {
                    upvar_tys: ref tys, ..
                }) => {
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
            Ok(self.normalize(field.ty(tcx, substs)))
        } else {
            Err(FieldAccessError::OutOfRange { field_count: variant.fields.len() })
        }
    }

    fn normalize(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
        let infcx = self.infcx();
        let mut selcx = traits::SelectionContext::new(infcx);
        let cause = traits::ObligationCause::misc(self.last_span, 0);
        let traits::Normalized { value: ty, obligations } =
            traits::normalize(&mut selcx, cause, &ty);

        debug!("normalize: ty={:?} obligations={:?}",
               ty,
               obligations);

        let mut fulfill_cx = &mut self.cx.fulfillment_cx;
        for obligation in obligations {
            fulfill_cx.register_predicate_obligation(infcx, obligation);
        }

        ty
    }
}

pub struct TypeChecker<'a, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'tcx>,
    fulfillment_cx: traits::FulfillmentContext<'tcx>,
    last_span: Span
}

impl<'a, 'tcx> TypeChecker<'a, 'tcx> {
    fn new(infcx: &'a InferCtxt<'a, 'tcx>) -> Self {
        TypeChecker {
            infcx: infcx,
            fulfillment_cx: traits::FulfillmentContext::new(),
            last_span: DUMMY_SP
        }
    }

    fn mk_subty(&self, span: Span, sup: Ty<'tcx>, sub: Ty<'tcx>)
                -> infer::UnitResult<'tcx>
    {
        infer::mk_subty(self.infcx, false, infer::TypeOrigin::Misc(span),
                        sup, sub)
            // FIXME(#32730) propagate obligations
            .map(|InferOk { obligations, .. }| assert!(obligations.is_empty()))
    }

    fn mk_eqty(&self, span: Span, a: Ty<'tcx>, b: Ty<'tcx>)
                -> infer::UnitResult<'tcx>
    {
        infer::mk_eqty(self.infcx, false, infer::TypeOrigin::Misc(span),
                       a, b)
            // FIXME(#32730) propagate obligations
            .map(|InferOk { obligations, .. }| assert!(obligations.is_empty()))
    }

    fn tcx(&self) -> &'a TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn check_stmt(&mut self, mir: &Mir<'tcx>, stmt: &Statement<'tcx>) {
        debug!("check_stmt: {:?}", stmt);
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(ref lv, ref rv) => {
                let lv_ty = mir.lvalue_ty(tcx, lv).to_ty(tcx);
                let rv_ty = mir.rvalue_ty(tcx, rv);
                if let Some(rv_ty) = rv_ty {
                    if let Err(terr) = self.mk_subty(self.last_span, rv_ty, lv_ty) {
                        span_mirbug!(self, stmt, "bad assignment ({:?} = {:?}): {:?}",
                                     lv_ty, rv_ty, terr);
                    }
                }

                // FIXME: rvalue with undeterminable type - e.g. inline
                // asm.
            }
        }
    }

    fn check_terminator(&self,
                        mir: &Mir<'tcx>,
                        term: &Terminator<'tcx>) {
        debug!("check_terminator: {:?}", term);
        let tcx = self.tcx();
        match term.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::Drop { .. } => {
                // no checks needed for these
            }

            TerminatorKind::If { ref cond, .. } => {
                let cond_ty = mir.operand_ty(tcx, cond);
                match cond_ty.sty {
                    ty::TyBool => {}
                    _ => {
                        span_mirbug!(self, term, "bad If ({:?}, not bool", cond_ty);
                    }
                }
            }
            TerminatorKind::SwitchInt { ref discr, switch_ty, .. } => {
                let discr_ty = mir.lvalue_ty(tcx, discr).to_ty(tcx);
                if let Err(terr) = self.mk_subty(self.last_span, discr_ty, switch_ty) {
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
                let discr_ty = mir.lvalue_ty(tcx, discr).to_ty(tcx);
                match discr_ty.sty {
                    ty::TyEnum(def, _)
                        if def == adt_def && adt_def.variants.len() == targets.len()
                        => {},
                    _ => {
                        span_mirbug!(self, term, "bad Switch ({:?} on {:?})",
                                     adt_def, discr_ty);
                    }
                }
            }
            TerminatorKind::Call { ref func, ref args, ref destination, .. } => {
                let func_ty = mir.operand_ty(tcx, func);
                debug!("check_terminator: call, func_ty={:?}", func_ty);
                let func_ty = match func_ty.sty {
                    ty::TyFnDef(_, _, func_ty) | ty::TyFnPtr(func_ty) => func_ty,
                    _ => {
                        span_mirbug!(self, term, "call to non-function {:?}", func_ty);
                        return;
                    }
                };
                let sig = tcx.erase_late_bound_regions(&func_ty.sig);
                self.check_call_dest(mir, term, &sig, destination);

                if self.is_box_free(func) {
                    self.check_box_free_inputs(mir, term, &sig, args);
                } else {
                    self.check_call_inputs(mir, term, &sig, args);
                }
            }
        }
    }

    fn check_call_dest(&self,
                       mir: &Mir<'tcx>,
                       term: &Terminator<'tcx>,
                       sig: &ty::FnSig<'tcx>,
                       destination: &Option<(Lvalue<'tcx>, BasicBlock)>) {
        let tcx = self.tcx();
        match (destination, sig.output) {
            (&Some(..), ty::FnDiverging) => {
                span_mirbug!(self, term, "call to diverging function {:?} with dest", sig);
            }
            (&Some((ref dest, _)), ty::FnConverging(ty)) => {
                let dest_ty = mir.lvalue_ty(tcx, dest).to_ty(tcx);
                if let Err(terr) = self.mk_subty(self.last_span, ty, dest_ty) {
                    span_mirbug!(self, term,
                                 "call dest mismatch ({:?} <- {:?}): {:?}",
                                 dest_ty, ty, terr);
                }
            }
            (&None, ty::FnDiverging) => {}
            (&None, ty::FnConverging(..)) => {
                span_mirbug!(self, term, "call to converging function {:?} w/o dest", sig);
             }
        }
    }

    fn check_call_inputs(&self,
                         mir: &Mir<'tcx>,
                         term: &Terminator<'tcx>,
                         sig: &ty::FnSig<'tcx>,
                         args: &[Operand<'tcx>])
    {
        debug!("check_call_inputs({:?}, {:?})", sig, args);
        if args.len() < sig.inputs.len() ||
           (args.len() > sig.inputs.len() && !sig.variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }
        for (n, (fn_arg, op_arg)) in sig.inputs.iter().zip(args).enumerate() {
            let op_arg_ty = mir.operand_ty(self.tcx(), op_arg);
            if let Err(terr) = self.mk_subty(self.last_span, op_arg_ty, fn_arg) {
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

    fn check_box_free_inputs(&self,
                             mir: &Mir<'tcx>,
                             term: &Terminator<'tcx>,
                             sig: &ty::FnSig<'tcx>,
                             args: &[Operand<'tcx>])
    {
        debug!("check_box_free_inputs");

        // box_free takes a Box as a pointer. Allow for that.

        if sig.inputs.len() != 1 {
            span_mirbug!(self, term, "box_free should take 1 argument");
            return;
        }

        let pointee_ty = match sig.inputs[0].sty {
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

        let arg_ty = match mir.operand_ty(self.tcx(), &args[0]).sty {
            ty::TyRawPtr(mt) => mt.ty,
            ty::TyBox(ty) => ty,
            _ => {
                span_mirbug!(self, term, "box_free called with bad arg ty");
                return;
            }
        };

        if let Err(terr) = self.mk_subty(self.last_span, arg_ty, pointee_ty) {
            span_mirbug!(self, term, "bad box_free arg ({:?} <- {:?}): {:?}",
                         pointee_ty, arg_ty, terr);
        }
    }

    fn typeck_mir(&mut self, mir: &Mir<'tcx>) {
        self.last_span = mir.span;
        debug!("run_on_mir: {:?}", mir.span);
        for block in &mir.basic_blocks {
            for stmt in &block.statements {
                if stmt.span != DUMMY_SP {
                    self.last_span = stmt.span;
                }
                self.check_stmt(mir, stmt);
            }

            if let Some(ref terminator) = block.terminator {
                self.check_terminator(mir, terminator);
            }
        }
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
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, id: NodeId, mir: &mut Mir<'tcx>) {
        if tcx.sess.err_count() > 0 {
            // compiling a broken program can obviously result in a
            // broken MIR, so try not to report duplicate errors.
            return;
        }
        let def_id = tcx.map.local_def_id(id);
        let _task = tcx.dep_graph.in_task(DepNode::MirTypeck(def_id));
        let param_env = ty::ParameterEnvironment::for_item(tcx, id);
        let infcx = infer::new_infer_ctxt(tcx,
                                          &tcx.tables,
                                          Some(param_env),
                                          ProjectionMode::AnyFinal);
        let mut checker = TypeChecker::new(&infcx);
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
    }
}

impl Pass for TypeckMir {}
