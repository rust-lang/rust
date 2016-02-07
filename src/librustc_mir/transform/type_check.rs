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

use rustc::middle::infer;
use rustc::middle::ty::{self, Ty};
use rustc::middle::ty::fold::TypeFoldable;
use rustc::mir::repr::*;
use rustc::mir::transform::MirPass;
use rustc::mir::visit::{self, Visitor};

use syntax::codemap::{Span, DUMMY_SP};
use std::fmt;

macro_rules! span_mirbug {
    ($context:expr, $elem:expr, $($message:tt)*) => ({
        $context.tcx().sess.span_warn(
            $context.last_span,
            &format!("broken MIR ({:?}): {:?}", $elem, format!($($message)*))
        )
    })
}

/// Verifies that MIR types are sane to not crash further
/// checks.
struct TypeVerifier<'a, 'tcx: 'a> {
    infcx: &'a infer::InferCtxt<'a, 'tcx>,
    mir: &'a Mir<'tcx>,
    last_span: Span,
    errors_reported: bool
}

impl<'a, 'tcx> Visitor<'tcx> for TypeVerifier<'a, 'tcx> {
    fn visit_span(&mut self, span: &Span) {
        if *span != DUMMY_SP {
            self.last_span = *span;
        }
    }

    fn visit_lvalue(&mut self, lvalue: &Lvalue<'tcx>, context: visit::LvalueContext) {
        self.super_lvalue(lvalue, context);
        let lv_ty = self.mir.lvalue_ty(self.tcx(), lvalue).to_ty(self.tcx());
        self.sanitize_type(lvalue, lv_ty);
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
        self.super_mir(mir);
    }
}

impl<'a, 'tcx> TypeVerifier<'a, 'tcx> {
    fn new(infcx: &'a infer::InferCtxt<'a, 'tcx>, mir: &'a Mir<'tcx>) -> Self {
        TypeVerifier {
            infcx: infcx,
            mir: mir,
            last_span: mir.span,
            errors_reported: false
        }
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn sanitize_type(&mut self, parent: &fmt::Debug, ty: Ty<'tcx>) {
        if !(ty.needs_infer() || ty.has_escaping_regions()) {
            return;
        }
        span_mirbug!(self, parent, "bad type {:?}", ty);
        self.errors_reported = true;
    }
}

pub struct TypeckMir<'a, 'tcx: 'a> {
    infcx: &'a infer::InferCtxt<'a, 'tcx>,
    last_span: Span
}

impl<'a, 'tcx> TypeckMir<'a, 'tcx> {
    pub fn new(infcx: &'a infer::InferCtxt<'a, 'tcx>) -> Self {
        TypeckMir {
            infcx: infcx,
            last_span: DUMMY_SP
        }
    }

    fn tcx(&self) -> &'a ty::ctxt<'tcx> {
        self.infcx.tcx
    }

    fn check_stmt(&mut self, mir: &Mir<'tcx>, stmt: &Statement<'tcx>) {
        debug!("check_stmt: {:?}", stmt);
        let tcx = self.tcx();
        match stmt.kind {
            StatementKind::Assign(ref lv, ref rv) => {
                match lv {
                    &Lvalue::ReturnPointer if mir.return_ty == ty::FnDiverging => {
                        // HACK: buggy writes
                        return;
                    }
                    _ => {}
                }

                let lv_ty = mir.lvalue_ty(tcx, lv).to_ty(tcx);
                let rv_ty = mir.rvalue_ty(tcx, rv);
                if let Some(rv_ty) = rv_ty {
                    if let Err(terr) = infer::can_mk_subty(self.infcx, rv_ty, lv_ty) {
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
        match *term {
            Terminator::Goto { .. } |
            Terminator::Resume |
            Terminator::Return |
            Terminator::Drop { .. } => {}
            Terminator::If { ref cond, .. } => {
                let cond_ty = mir.operand_ty(tcx, cond);
                match cond_ty.sty {
                    ty::TyBool => {}
                    _ => {
                        span_mirbug!(self, term, "bad If ({:?}, not bool", cond_ty);
                    }
                }
            }
            Terminator::SwitchInt { ref discr, switch_ty, .. } => {
                let discr_ty = mir.lvalue_ty(tcx, discr).to_ty(tcx);
                if let Err(terr) = infer::can_mk_subty(self.infcx, discr_ty, switch_ty) {
                    span_mirbug!(self, term, "bad SwitchInt ({:?} on {:?}): {:?}",
                                 switch_ty, discr_ty, terr);
                }
            }
            Terminator::Switch { ref discr, adt_def, .. } => {
                let discr_ty = mir.lvalue_ty(tcx, discr).to_ty(tcx);
                match discr_ty.sty {
                    ty::TyEnum(def, _) if def == adt_def => {},
                    _ => {
                        span_mirbug!(self, term, "bad Switch ({:?} on {:?})",
                                     adt_def, discr_ty);
                    }
                }
            }
            Terminator::Call { ref func, ref args, ref destination, .. } => {
                let func_ty = mir.operand_ty(tcx, func);
                debug!("check_terminator: call, func_ty={:?}", func_ty);
                let func_ty = match func_ty.sty {
                    ty::TyBareFn(_, func_ty) => func_ty,
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
                if let Err(terr) = infer::can_mk_subty(self.infcx, ty, dest_ty) {
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
        if sig.inputs.len() > args.len() ||
           (sig.inputs.len() < args.len() && !sig.variadic) {
            span_mirbug!(self, term, "call to {:?} with wrong # of args", sig);
        }
        for (n, (fn_arg, op_arg)) in sig.inputs.iter().zip(args).enumerate() {
            let op_arg_ty = mir.operand_ty(self.tcx(), op_arg);
            if let Err(terr) = infer::can_mk_subty(self.infcx, op_arg_ty, fn_arg) {
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

        if let Err(terr) = infer::can_mk_subty(self.infcx, arg_ty, pointee_ty) {
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
}

impl<'a, 'tcx> MirPass for TypeckMir<'a, 'tcx> {
    fn run_on_mir<'tcx_>(&mut self,
                         mir: &mut Mir<'tcx_>,
                         _tcx: &ty::ctxt<'tcx_>) {
        // FIXME: pass param_env to run_on_mir
        let mir: &mut Mir<'tcx> = unsafe { ::std::mem::transmute(mir) };
        let mut type_verifier = TypeVerifier::new(self.infcx, mir);
        type_verifier.visit_mir(mir);

        if type_verifier.errors_reported {
            return;
        }

        self.typeck_mir(mir);
    }
}
