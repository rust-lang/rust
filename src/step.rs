//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use std::cell::Ref;

use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc::mir;
use rustc::ty::layout::Layout;
use rustc::ty::{subst, self};

use error::{EvalResult, EvalError};
use eval_context::{EvalContext, StackPopCleanup, MirRef};
use lvalue::{Global, GlobalId, Lvalue};
use syntax::codemap::Span;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    pub fn inc_step_counter_and_check_limit(&mut self, n: u64) -> EvalResult<'tcx> {
        self.steps_remaining = self.steps_remaining.saturating_sub(n);
        if self.steps_remaining > 0 {
            Ok(())
        } else {
            Err(EvalError::ExecutionTimeLimitReached)
        }
    }

    /// Returns true as long as there are more things to do.
    pub fn step(&mut self) -> EvalResult<'tcx, bool> {
        // see docs on the `Memory::packed` field for why we do this
        self.memory.clear_packed();
        self.inc_step_counter_and_check_limit(1)?;
        if self.stack.is_empty() {
            return Ok(false);
        }

        let block = self.frame().block;
        let stmt_id = self.frame().stmt;
        let mir = self.mir();
        let basic_block = &mir.basic_blocks()[block];

        if let Some(stmt) = basic_block.statements.get(stmt_id) {
            let mut new = Ok(0);
            ConstantExtractor {
                span: stmt.source_info.span,
                substs: self.substs(),
                def_id: self.frame().def_id,
                ecx: self,
                mir: Ref::clone(&mir),
                new_constants: &mut new,
            }.visit_statement(block, stmt, mir::Location { block, statement_index: stmt_id });
            if new? == 0 {
                self.statement(stmt)?;
            }
            // if ConstantExtractor added new frames, we don't execute anything here
            // but await the next call to step
            return Ok(true);
        }

        let terminator = basic_block.terminator();
        let mut new = Ok(0);
        ConstantExtractor {
            span: terminator.source_info.span,
            substs: self.substs(),
            def_id: self.frame().def_id,
            ecx: self,
            mir: Ref::clone(&mir),
            new_constants: &mut new,
        }.visit_terminator(block, terminator, mir::Location { block, statement_index: stmt_id });
        if new? == 0 {
            self.terminator(terminator)?;
        }
        // if ConstantExtractor added new frames, we don't execute anything here
        // but await the next call to step
        Ok(true)
    }

    fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", stmt);

        use rustc::mir::StatementKind::*;
        match stmt.kind {
            Assign(ref lvalue, ref rvalue) => self.eval_rvalue_into_lvalue(rvalue, lvalue)?,

            SetDiscriminant { ref lvalue, variant_index } => {
                let dest = self.eval_lvalue(lvalue)?;
                let dest_ty = self.lvalue_ty(lvalue);
                let dest_layout = self.type_layout(dest_ty)?;

                match *dest_layout {
                    Layout::General { discr, ref variants, .. } => {
                        let discr_size = discr.size().bytes();
                        let discr_offset = variants[variant_index].offsets[0].bytes();

                        // FIXME(solson)
                        let dest = self.force_allocation(dest)?;
                        let discr_dest = (dest.to_ptr()).offset(discr_offset);

                        self.memory.write_uint(discr_dest, variant_index as u128, discr_size)?;
                    }

                    Layout::RawNullablePointer { nndiscr, .. } => {
                        use value::PrimVal;
                        if variant_index as u64 != nndiscr {
                            self.write_primval(dest, PrimVal::Bytes(0), dest_ty)?;
                        }
                    }

                    _ => bug!("SetDiscriminant on {} represented as {:#?}", dest_ty, dest_layout),
                }
            }

            // Miri can safely ignore these. Only translation needs it.
            StorageLive(_) |
            StorageDead(_) => {}

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}
        }

        self.frame_mut().stmt += 1;
        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", terminator.kind);
        self.eval_terminator(terminator)?;
        if !self.stack.is_empty() {
            trace!("// {:?}", self.frame().block);
        }
        Ok(())
    }
}

// WARNING: make sure that any methods implemented on this type don't ever access ecx.stack
// this includes any method that might access the stack
// basically don't call anything other than `load_mir`, `alloc_ptr`, `push_stack_frame`
// The reason for this is, that `push_stack_frame` modifies the stack out of obvious reasons
struct ConstantExtractor<'a, 'b: 'a, 'tcx: 'b> {
    span: Span,
    ecx: &'a mut EvalContext<'b, 'tcx>,
    mir: MirRef<'tcx>,
    def_id: DefId,
    substs: &'tcx subst::Substs<'tcx>,
    new_constants: &'a mut EvalResult<'tcx, u64>,
}

impl<'a, 'b, 'tcx> ConstantExtractor<'a, 'b, 'tcx> {
    fn global_item(&mut self, def_id: DefId, substs: &'tcx subst::Substs<'tcx>, span: Span, immutable: bool) {
        let cid = GlobalId { def_id, substs, promoted: None };
        if self.ecx.globals.contains_key(&cid) {
            return;
        }
        self.try(|this| {
            let mir = this.ecx.load_mir(def_id)?;
            this.ecx.globals.insert(cid, Global::uninitialized(mir.return_ty));
            let cleanup = if immutable && !mir.return_ty.type_contents(this.ecx.tcx).interior_unsafe() {
                StackPopCleanup::Freeze
            } else {
                StackPopCleanup::None
            };
            this.ecx.push_stack_frame(def_id, span, mir, substs, Lvalue::Global(cid), cleanup, Vec::new())
        });
    }
    fn try<F: FnOnce(&mut Self) -> EvalResult<'tcx>>(&mut self, f: F) {
        if let Ok(ref mut n) = *self.new_constants {
            *n += 1;
        } else {
            return;
        }
        if let Err(e) = f(self) {
            *self.new_constants = Err(e);
        }
    }
}

impl<'a, 'b, 'tcx> Visitor<'tcx> for ConstantExtractor<'a, 'b, 'tcx> {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>, location: mir::Location) {
        self.super_constant(constant, location);
        match constant.literal {
            // already computed by rustc
            mir::Literal::Value { .. } => {}
            mir::Literal::Item { def_id, substs } => {
                if let ty::TyFnDef(..) = constant.ty.sty {
                    // No need to do anything here,
                    // because the type is the actual function, not the signature of the function.
                    // Thus we can simply create a zero sized allocation in `evaluate_operand`
                } else {
                    self.global_item(def_id, substs, constant.span, true);
                }
            },
            mir::Literal::Promoted { index } => {
                let cid = GlobalId {
                    def_id: self.def_id,
                    substs: self.substs,
                    promoted: Some(index),
                };
                if self.ecx.globals.contains_key(&cid) {
                    return;
                }
                let mir = Ref::clone(&self.mir);
                let mir = Ref::map(mir, |mir| &mir.promoted[index]);
                self.try(|this| {
                    let ty = this.ecx.monomorphize(mir.return_ty, this.substs);
                    this.ecx.globals.insert(cid, Global::uninitialized(ty));
                    this.ecx.push_stack_frame(this.def_id,
                                              constant.span,
                                              mir,
                                              this.substs,
                                              Lvalue::Global(cid),
                                              StackPopCleanup::Freeze,
                                              Vec::new())
                });
            }
        }
    }

    fn visit_lvalue(
        &mut self,
        lvalue: &mir::Lvalue<'tcx>,
        context: LvalueContext<'tcx>,
        location: mir::Location
    ) {
        self.super_lvalue(lvalue, context, location);
        if let mir::Lvalue::Static(def_id) = *lvalue {
            let substs = self.ecx.tcx.intern_substs(&[]);
            let span = self.span;
            if let Some(node_item) = self.ecx.tcx.hir.get_if_local(def_id) {
                if let hir::map::Node::NodeItem(&hir::Item { ref node, .. }) = node_item {
                    if let hir::ItemStatic(_, m, _) = *node {
                        self.global_item(def_id, substs, span, m == hir::MutImmutable);
                        return;
                    } else {
                        bug!("static def id doesn't point to static");
                    }
                } else {
                    bug!("static def id doesn't point to item");
                }
            } else {
                let def = self.ecx.tcx.sess.cstore.describe_def(def_id).expect("static not found");
                if let hir::def::Def::Static(_, mutable) = def {
                    self.global_item(def_id, substs, span, !mutable);
                } else {
                    bug!("static found but isn't a static: {:?}", def);
                }
            }
        }
    }
}
