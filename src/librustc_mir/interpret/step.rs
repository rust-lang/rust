//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::mir::visit::{Visitor, LvalueContext};
use rustc::mir;
use rustc::traits::Reveal;
use rustc::ty;
use rustc::ty::layout::Layout;
use rustc::ty::subst::{Subst, Substs};

use error::{EvalResult, EvalError};
use eval_context::{EvalContext, StackPopCleanup};
use lvalue::{Global, GlobalId, Lvalue, ValidationCtx};
use value::{Value, PrimVal};
use syntax::codemap::Span;
use syntax::ast::Mutability;

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
        self.inc_step_counter_and_check_limit(1)?;
        if self.stack.is_empty() {
            return Ok(false);
        }

        let cur_frame = self.cur_frame();
        self.memory.set_cur_frame(cur_frame);

        let block = self.frame().block;
        let stmt_id = self.frame().stmt;
        let mir = self.mir();
        let basic_block = &mir.basic_blocks()[block];

        if let Some(stmt) = basic_block.statements.get(stmt_id) {
            let mut new = Ok(0);
            ConstantExtractor {
                span: stmt.source_info.span,
                instance: self.frame().instance,
                ecx: self,
                mir,
                new_constants: &mut new,
            }.visit_statement(block, stmt, mir::Location { block, statement_index: stmt_id });
            // if ConstantExtractor added new frames, we don't execute anything here
            // but await the next call to step
            if new? == 0 {
                self.statement(stmt)?;
            }
            return Ok(true);
        }

        let terminator = basic_block.terminator();
        let mut new = Ok(0);
        ConstantExtractor {
            span: terminator.source_info.span,
            instance: self.frame().instance,
            ecx: self,
            mir,
            new_constants: &mut new,
        }.visit_terminator(block, terminator, mir::Location { block, statement_index: stmt_id });
        // if ConstantExtractor added new frames, we don't execute anything here
        // but await the next call to step
        if new? == 0 {
            self.terminator(terminator)?;
        }
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
                    Layout::General { discr, .. } => {
                        let discr_size = discr.size().bytes();
                        let dest_ptr = self.force_allocation(dest)?.to_ptr()?;
                        self.memory.write_uint(dest_ptr, variant_index as u128, discr_size)?
                    }

                    Layout::RawNullablePointer { nndiscr, .. } => {
                        if variant_index as u64 != nndiscr {
                            self.write_null(dest, dest_ty)?;
                        }
                    }

                    Layout::StructWrappedNullablePointer { nndiscr, ref discrfield, .. } => {
                        if variant_index as u64 != nndiscr {
                            let (offset, ty) = self.nonnull_offset_and_ty(dest_ty, nndiscr, discrfield)?;
                            let nonnull = self.force_allocation(dest)?.to_ptr()?.offset(offset.bytes(), &self)?;
                            trace!("struct wrapped nullable pointer type: {}", ty);
                            // only the pointer part of a fat pointer is used for this space optimization
                            let discr_size = self.type_size(ty)?.expect("bad StructWrappedNullablePointer discrfield");
                            self.memory.write_uint(nonnull, 0, discr_size)?;
                        }
                    },

                    _ => bug!("SetDiscriminant on {} represented as {:#?}", dest_ty, dest_layout),
                }
            }

            // Mark locals as dead or alive.
            StorageLive(ref lvalue) | StorageDead(ref lvalue)=> {
                let (frame, local) = match self.eval_lvalue(lvalue)? {
                    Lvalue::Local{ frame, local } if self.cur_frame() == frame => (frame, local),
                    _ => return Err(EvalError::Unimplemented("Storage annotations must refer to locals of the topmost stack frame.".to_owned())) // FIXME maybe this should get its own error type
                };
                let old_val = match stmt.kind {
                    StorageLive(_) => self.stack[frame].storage_live(local)?,
                    StorageDead(_) =>  self.stack[frame].storage_dead(local)?,
                    _ => bug!("We already checked that we are a storage stmt")
                };
                self.deallocate_local(old_val)?;
            }

            // Validity checks.
            Validate(op, ref lvalues) => {
                for operand in lvalues {
                    // We need to monomorphize ty *without* erasing lifetimes
                    let ty = operand.ty.subst(self.tcx, self.substs());
                    // TODO: do we have to self.tcx.normalize_associated_type(&{ty}) ?  That however seems to erase lifetimes.
                    let lvalue = self.eval_lvalue(&operand.lval)?;
                    self.validate(lvalue, ty, ValidationCtx::new(op))?;
                }
            }
            EndRegion(ce) => {
                self.memory.locks_lifetime_ended(Some(ce));
            }

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            InlineAsm { .. } => return Err(EvalError::InlineAsm),
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
    mir: &'tcx mir::Mir<'tcx>,
    instance: ty::Instance<'tcx>,
    new_constants: &'a mut EvalResult<'tcx, u64>,
}

impl<'a, 'b, 'tcx> ConstantExtractor<'a, 'b, 'tcx> {
    fn global_item(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        span: Span,
        mutability: Mutability,
    ) {
        let instance = self.ecx.resolve_associated_const(def_id, substs);
        let cid = GlobalId { instance, promoted: None };
        if self.ecx.globals.contains_key(&cid) {
            return;
        }
        if self.ecx.tcx.has_attr(def_id, "linkage") {
            trace!("Initializing an extern global with NULL");
            self.ecx.globals.insert(cid, Global::initialized(self.ecx.tcx.type_of(def_id), Value::ByVal(PrimVal::Bytes(0)), mutability));
            return;
        }
        self.try(|this| {
            let mir = this.ecx.load_mir(instance.def)?;
            this.ecx.globals.insert(cid, Global::uninitialized(mir.return_ty));
            let internally_mutable = !mir.return_ty.is_freeze(
                    this.ecx.tcx,
                    ty::ParamEnv::empty(Reveal::All),
                    span);
            let mutability = if mutability == Mutability::Mutable || internally_mutable {
                Mutability::Mutable
            } else {
                Mutability::Immutable
            };
            let cleanup = StackPopCleanup::MarkStatic(mutability);
            let name = ty::tls::with(|tcx| tcx.item_path_str(def_id));
            trace!("pushing stack frame for global: {}", name);
            this.ecx.push_stack_frame(
                instance,
                span,
                mir,
                Lvalue::Global(cid),
                cleanup,
            )
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
                self.global_item(def_id, substs, constant.span, Mutability::Immutable);
            },
            mir::Literal::Promoted { index } => {
                let cid = GlobalId {
                    instance: self.instance,
                    promoted: Some(index),
                };
                if self.ecx.globals.contains_key(&cid) {
                    return;
                }
                let mir = &self.mir.promoted[index];
                self.try(|this| {
                    let ty = this.ecx.monomorphize(mir.return_ty, this.instance.substs);
                    this.ecx.globals.insert(cid, Global::uninitialized(ty));
                    trace!("pushing stack frame for {:?}", index);
                    this.ecx.push_stack_frame(this.instance,
                                              constant.span,
                                              mir,
                                              Lvalue::Global(cid),
                                              StackPopCleanup::MarkStatic(Mutability::Immutable),
                    )
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
        if let mir::Lvalue::Static(ref static_) = *lvalue {
            let def_id = static_.def_id;
            let substs = self.ecx.tcx.intern_substs(&[]);
            let span = self.span;
            if let Some(node_item) = self.ecx.tcx.hir.get_if_local(def_id) {
                if let hir::map::Node::NodeItem(&hir::Item { ref node, .. }) = node_item {
                    if let hir::ItemStatic(_, m, _) = *node {
                        self.global_item(def_id, substs, span, if m == hir::MutMutable { Mutability::Mutable } else { Mutability::Immutable });
                        return;
                    } else {
                        bug!("static def id doesn't point to static");
                    }
                } else {
                    bug!("static def id doesn't point to item");
                }
            } else {
                let def = self.ecx.tcx.describe_def(def_id).expect("static not found");
                if let hir::def::Def::Static(_, mutable) = def {
                    self.global_item(def_id, substs, span, if mutable { Mutability::Mutable } else { Mutability::Immutable });
                } else {
                    bug!("static found but isn't a static: {:?}", def);
                }
            }
        }
    }
}
