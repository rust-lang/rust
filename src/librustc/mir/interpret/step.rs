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
use rustc::ty::subst::Substs;
use rustc::middle::const_val::ConstVal;

use super::{EvalResult, EvalContext, StackPopCleanup, PtrAndAlign, GlobalId, Lvalue,
            MemoryKind, Machine, PrimVal};

use syntax::codemap::Span;
use syntax::ast::Mutability;

impl<'a, 'tcx, M: Machine<'tcx>> EvalContext<'a, 'tcx, M> {
    pub fn inc_step_counter_and_check_limit(&mut self, n: u64) -> EvalResult<'tcx> {
        self.steps_remaining = self.steps_remaining.saturating_sub(n);
        if self.steps_remaining > 0 {
            Ok(())
        } else {
            err!(ExecutionTimeLimitReached)
        }
    }

    /// Returns true as long as there are more things to do.
    pub fn step(&mut self) -> EvalResult<'tcx, bool> {
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
                instance: self.frame().instance,
                ecx: self,
                mir,
                new_constants: &mut new,
            }.visit_statement(
                block,
                stmt,
                mir::Location {
                    block,
                    statement_index: stmt_id,
                },
            );
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
        }.visit_terminator(
            block,
            terminator,
            mir::Location {
                block,
                statement_index: stmt_id,
            },
        );
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

        // Some statements (e.g. box) push new stack frames.  We have to record the stack frame number
        // *before* executing the statement.
        let frame_idx = self.cur_frame();

        match stmt.kind {
            Assign(ref lvalue, ref rvalue) => self.eval_rvalue_into_lvalue(rvalue, lvalue)?,

            SetDiscriminant {
                ref lvalue,
                variant_index,
            } => {
                let dest = self.eval_lvalue(lvalue)?;
                let dest_ty = self.lvalue_ty(lvalue);
                let dest_layout = self.type_layout(dest_ty)?;

                match *dest_layout {
                    Layout::General { discr, .. } => {
                        let discr_size = discr.size().bytes();
                        let dest_ptr = self.force_allocation(dest)?.to_ptr()?;
                        self.memory.write_primval(
                            dest_ptr,
                            PrimVal::Bytes(variant_index as u128),
                            discr_size,
                            false
                        )?
                    }

                    Layout::RawNullablePointer { nndiscr, .. } => {
                        if variant_index as u64 != nndiscr {
                            self.write_null(dest, dest_ty)?;
                        }
                    }

                    Layout::StructWrappedNullablePointer {
                        nndiscr,
                        ref discrfield_source,
                        ..
                    } => {
                        if variant_index as u64 != nndiscr {
                            self.write_struct_wrapped_null_pointer(
                                dest_ty,
                                nndiscr,
                                discrfield_source,
                                dest,
                            )?;
                        }
                    }

                    _ => {
                        bug!(
                            "SetDiscriminant on {} represented as {:#?}",
                            dest_ty,
                            dest_layout
                        )
                    }
                }
            }

            // Mark locals as alive
            StorageLive(local) => {
                let old_val = self.frame_mut().storage_live(local)?;
                self.deallocate_local(old_val)?;
            }

            // Mark locals as dead
            StorageDead(local) => {
                let old_val = self.frame_mut().storage_dead(local)?;
                self.deallocate_local(old_val)?;
            }

            // Validity checks.
            Validate(op, ref lvalues) => {
                for operand in lvalues {
                    self.validation_op(op, operand)?;
                }
            }
            EndRegion(ce) => {
                self.end_region(Some(ce))?;
            }

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            InlineAsm { .. } => return err!(InlineAsm),
        }

        self.stack[frame_idx].stmt += 1;
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

    /// returns `true` if a stackframe was pushed
    fn global_item(
        &mut self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        span: Span,
        mutability: Mutability,
    ) -> EvalResult<'tcx, bool> {
        let instance = self.resolve_associated_const(def_id, substs);
        let cid = GlobalId {
            instance,
            promoted: None,
        };
        if self.globals.contains_key(&cid) {
            return Ok(false);
        }
        if self.tcx.has_attr(def_id, "linkage") {
            M::global_item_with_linkage(self, cid.instance, mutability)?;
            return Ok(false);
        }
        let mir = self.load_mir(instance.def)?;
        let size = self.type_size_with_substs(mir.return_ty, substs)?.expect(
            "unsized global",
        );
        let align = self.type_align_with_substs(mir.return_ty, substs)?;
        let ptr = self.memory.allocate(
            size,
            align,
            MemoryKind::UninitializedStatic,
        )?;
        let aligned = !self.is_packed(mir.return_ty)?;
        self.globals.insert(
            cid,
            PtrAndAlign {
                ptr: ptr.into(),
                aligned,
            },
        );
        let internally_mutable = !mir.return_ty.is_freeze(
            self.tcx,
            ty::ParamEnv::empty(Reveal::All),
            span,
        );
        let mutability = if mutability == Mutability::Mutable || internally_mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        };
        let cleanup = StackPopCleanup::MarkStatic(mutability);
        let name = ty::tls::with(|tcx| tcx.item_path_str(def_id));
        trace!("pushing stack frame for global: {}", name);
        self.push_stack_frame(
            instance,
            span,
            mir,
            Lvalue::from_ptr(ptr),
            cleanup,
        )?;
        Ok(true)
    }
}

// WARNING: This code pushes new stack frames.  Make sure that any methods implemented on this
// type don't ever access ecx.stack[ecx.cur_frame()], as that will change. This includes, e.g.,
// using the current stack frame's substitution.
// Basically don't call anything other than `load_mir`, `alloc_ptr`, `push_stack_frame`.
struct ConstantExtractor<'a, 'b: 'a, 'tcx: 'b, M: Machine<'tcx> + 'a> {
    span: Span,
    ecx: &'a mut EvalContext<'b, 'tcx, M>,
    mir: &'tcx mir::Mir<'tcx>,
    instance: ty::Instance<'tcx>,
    new_constants: &'a mut EvalResult<'tcx, u64>,
}

impl<'a, 'b, 'tcx, M: Machine<'tcx>> ConstantExtractor<'a, 'b, 'tcx, M> {
    fn try<F: FnOnce(&mut Self) -> EvalResult<'tcx, bool>>(&mut self, f: F) {
        // previous constant errored
        let n = match *self.new_constants {
            Ok(n) => n,
            Err(_) => return,
        };
        match f(self) {
            // everything ok + a new stackframe
            Ok(true) => *self.new_constants = Ok(n + 1),
            // constant correctly evaluated, but no new stackframe
            Ok(false) => {}
            // constant eval errored
            Err(err) => *self.new_constants = Err(err),
        }
    }
}

impl<'a, 'b, 'tcx, M: Machine<'tcx>> Visitor<'tcx> for ConstantExtractor<'a, 'b, 'tcx, M> {
    fn visit_constant(&mut self, constant: &mir::Constant<'tcx>, location: mir::Location) {
        self.super_constant(constant, location);
        match constant.literal {
            // already computed by rustc
            mir::Literal::Value { value: &ty::Const { val: ConstVal::Unevaluated(def_id, substs), .. } } => {
                self.try(|this| {
                    this.ecx.global_item(
                        def_id,
                        substs,
                        constant.span,
                        Mutability::Immutable,
                    )
                });
            }
            mir::Literal::Value { .. } => {}
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
                    let size = this.ecx
                        .type_size_with_substs(mir.return_ty, this.instance.substs)?
                        .expect("unsized global");
                    let align = this.ecx.type_align_with_substs(
                        mir.return_ty,
                        this.instance.substs,
                    )?;
                    let ptr = this.ecx.memory.allocate(
                        size,
                        align,
                        MemoryKind::UninitializedStatic,
                    )?;
                    let aligned = !this.ecx.is_packed(mir.return_ty)?;
                    this.ecx.globals.insert(
                        cid,
                        PtrAndAlign {
                            ptr: ptr.into(),
                            aligned,
                        },
                    );
                    trace!("pushing stack frame for {:?}", index);
                    this.ecx.push_stack_frame(
                        this.instance,
                        constant.span,
                        mir,
                        Lvalue::from_ptr(ptr),
                        StackPopCleanup::MarkStatic(Mutability::Immutable),
                    )?;
                    Ok(true)
                });
            }
        }
    }

    fn visit_lvalue(
        &mut self,
        lvalue: &mir::Lvalue<'tcx>,
        context: LvalueContext<'tcx>,
        location: mir::Location,
    ) {
        self.super_lvalue(lvalue, context, location);
        if let mir::Lvalue::Static(ref static_) = *lvalue {
            let def_id = static_.def_id;
            let substs = self.ecx.tcx.intern_substs(&[]);
            let span = self.span;
            if let Some(node_item) = self.ecx.tcx.hir.get_if_local(def_id) {
                if let hir::map::Node::NodeItem(&hir::Item { ref node, .. }) = node_item {
                    if let hir::ItemStatic(_, m, _) = *node {
                        self.try(|this| {
                            this.ecx.global_item(
                                def_id,
                                substs,
                                span,
                                if m == hir::MutMutable {
                                    Mutability::Mutable
                                } else {
                                    Mutability::Immutable
                                },
                            )
                        });
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
                    self.try(|this| {
                        this.ecx.global_item(
                            def_id,
                            substs,
                            span,
                            if mutable {
                                Mutability::Mutable
                            } else {
                                Mutability::Immutable
                            },
                        )
                    });
                } else {
                    bug!("static found but isn't a static: {:?}", def);
                }
            }
        }
    }
}
