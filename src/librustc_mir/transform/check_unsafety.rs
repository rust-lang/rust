// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::indexed_vec::IndexVec;

use rustc::ty::maps::Providers;
use rustc::ty::{self, TyCtxt};
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::lint::builtin::{SAFE_EXTERN_STATICS, UNUSED_UNSAFE};
use rustc::mir::*;
use rustc::mir::visit::{LvalueContext, Visitor};

use syntax::ast;

use std::rc::Rc;


pub struct UnsafetyChecker<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    visibility_scope_info: &'a IndexVec<VisibilityScope, VisibilityScopeInfo>,
    violations: Vec<UnsafetyViolation>,
    source_info: SourceInfo,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    used_unsafe: FxHashSet<ast::NodeId>,
    inherited_blocks: Vec<(ast::NodeId, bool)>,
}

impl<'a, 'gcx, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn new(mir: &'a Mir<'tcx>,
           visibility_scope_info: &'a IndexVec<VisibilityScope, VisibilityScopeInfo>,
           tcx: TyCtxt<'a, 'tcx, 'tcx>,
           param_env: ty::ParamEnv<'tcx>) -> Self {
        Self {
            mir,
            visibility_scope_info,
            violations: vec![],
            source_info: SourceInfo {
                span: mir.span,
                scope: ARGUMENT_VISIBILITY_SCOPE
            },
            tcx,
            param_env,
            used_unsafe: FxHashSet(),
            inherited_blocks: vec![],
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnsafetyChecker<'a, 'tcx> {
    fn visit_terminator(&mut self,
                        block: BasicBlock,
                        terminator: &Terminator<'tcx>,
                        location: Location)
    {
        self.source_info = terminator.source_info;
        match terminator.kind {
            TerminatorKind::Goto { .. } |
            TerminatorKind::SwitchInt { .. } |
            TerminatorKind::Drop { .. } |
            TerminatorKind::Yield { .. } |
            TerminatorKind::Assert { .. } |
            TerminatorKind::DropAndReplace { .. } |
            TerminatorKind::GeneratorDrop |
            TerminatorKind::Resume |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::FalseEdges { .. } => {
                // safe (at least as emitted during MIR construction)
            }

            TerminatorKind::Call { ref func, .. } => {
                let func_ty = func.ty(self.mir, self.tcx);
                let sig = func_ty.fn_sig(self.tcx);
                if let hir::Unsafety::Unsafe = sig.unsafety() {
                    self.require_unsafe("call to unsafe function")
                }
            }
        }
        self.super_terminator(block, terminator, location);
    }

    fn visit_statement(&mut self,
                       block: BasicBlock,
                       statement: &Statement<'tcx>,
                       location: Location)
    {
        self.source_info = statement.source_info;
        match statement.kind {
            StatementKind::Assign(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(..) |
            StatementKind::StorageDead(..) |
            StatementKind::EndRegion(..) |
            StatementKind::Validate(..) |
            StatementKind::Nop => {
                // safe (at least as emitted during MIR construction)
            }

            StatementKind::InlineAsm { .. } => {
                self.require_unsafe("use of inline assembly")
            },
        }
        self.super_statement(block, statement, location);
    }

    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location)
    {
        if let &Rvalue::Aggregate(box ref aggregate, _) = rvalue {
            match aggregate {
                &AggregateKind::Array(..) |
                &AggregateKind::Tuple |
                &AggregateKind::Adt(..) => {}
                &AggregateKind::Closure(def_id, _) |
                &AggregateKind::Generator(def_id, _, _) => {
                    let UnsafetyCheckResult {
                        violations, unsafe_blocks
                    } = self.tcx.unsafety_check_result(def_id);
                    self.inherited_blocks.extend(unsafe_blocks.iter().cloned());
                    self.register_violations(&violations, &unsafe_blocks);
                }
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_lvalue(&mut self,
                    lvalue: &Lvalue<'tcx>,
                    context: LvalueContext<'tcx>,
                    location: Location) {
        match lvalue {
            &Lvalue::Projection(box Projection {
                ref base, ref elem
            }) => {
                let old_source_info = self.source_info;
                if let &Lvalue::Local(local) = base {
                    if self.mir.local_decls[local].internal {
                        // Internal locals are used in the `move_val_init` desugaring.
                        // We want to check unsafety against the source info of the
                        // desugaring, rather than the source info of the RHS.
                        self.source_info = self.mir.local_decls[local].source_info;
                    }
                }
                let base_ty = base.ty(self.mir, self.tcx).to_ty(self.tcx);
                match base_ty.sty {
                    ty::TyRawPtr(..) => {
                        self.require_unsafe("dereference of raw pointer")
                    }
                    ty::TyAdt(adt, _) if adt.is_union() => {
                        if context == LvalueContext::Store ||
                            context == LvalueContext::Drop
                        {
                            let elem_ty = match elem {
                                &ProjectionElem::Field(_, ty) => ty,
                                _ => span_bug!(
                                    self.source_info.span,
                                    "non-field projection {:?} from union?",
                                    lvalue)
                            };
                            if elem_ty.moves_by_default(self.tcx, self.param_env,
                                                        self.source_info.span) {
                                self.require_unsafe(
                                    "assignment to non-`Copy` union field")
                            } else {
                                // write to non-move union, safe
                            }
                        } else {
                            self.require_unsafe("access to union field")
                        }
                    }
                    _ => {}
                }
                self.source_info = old_source_info;
            }
            &Lvalue::Local(..) => {
                // locals are safe
            }
            &Lvalue::Static(box Static { def_id, ty: _ }) => {
                if self.tcx.is_static_mut(def_id) {
                    self.require_unsafe("use of mutable static");
                } else if self.tcx.is_foreign_item(def_id) {
                    let source_info = self.source_info;
                    let lint_root =
                        self.visibility_scope_info[source_info.scope].lint_root;
                    self.register_violations(&[UnsafetyViolation {
                        source_info,
                        description: "use of extern static",
                        lint_node_id: Some(lint_root)
                    }], &[]);
                }
            }
        }
        self.super_lvalue(lvalue, context, location);
    }
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn require_unsafe(&mut self,
                      description: &'static str)
    {
        let source_info = self.source_info;
        self.register_violations(&[UnsafetyViolation {
            source_info, description, lint_node_id: None
        }], &[]);
    }

    fn register_violations(&mut self,
                           violations: &[UnsafetyViolation],
                           unsafe_blocks: &[(ast::NodeId, bool)]) {
        let within_unsafe = match self.visibility_scope_info[self.source_info.scope].safety {
            Safety::Safe => {
                for violation in violations {
                    if !self.violations.contains(violation) {
                        self.violations.push(violation.clone())
                    }
                }

                false
            }
            Safety::BuiltinUnsafe | Safety::FnUnsafe => true,
            Safety::ExplicitUnsafe(node_id) => {
                if !violations.is_empty() {
                    self.used_unsafe.insert(node_id);
                }
                true
            }
        };
        self.inherited_blocks.extend(unsafe_blocks.iter().map(|&(node_id, is_used)| {
            (node_id, is_used && !within_unsafe)
        }));
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        unsafety_check_result,
        ..*providers
    };
}

struct UnusedUnsafeVisitor<'a> {
    used_unsafe: &'a FxHashSet<ast::NodeId>,
    unsafe_blocks: &'a mut Vec<(ast::NodeId, bool)>,
}

impl<'a, 'tcx> hir::intravisit::Visitor<'tcx> for UnusedUnsafeVisitor<'a> {
    fn nested_visit_map<'this>(&'this mut self) ->
        hir::intravisit::NestedVisitorMap<'this, 'tcx>
    {
        hir::intravisit::NestedVisitorMap::None
    }

    fn visit_block(&mut self, block: &'tcx hir::Block) {
        hir::intravisit::walk_block(self, block);

        if let hir::UnsafeBlock(hir::UserProvided) = block.rules {
            self.unsafe_blocks.push((block.id, self.used_unsafe.contains(&block.id)));
        }
    }
}

fn check_unused_unsafe<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                 def_id: DefId,
                                 used_unsafe: &FxHashSet<ast::NodeId>,
                                 unsafe_blocks: &'a mut Vec<(ast::NodeId, bool)>)
{
    let body_id =
        tcx.hir.as_local_node_id(def_id).and_then(|node_id| {
            tcx.hir.maybe_body_owned_by(node_id)
        });

    let body_id = match body_id {
        Some(body) => body,
        None => {
            debug!("check_unused_unsafe({:?}) - no body found", def_id);
            return
        }
    };
    let body = tcx.hir.body(body_id);
    debug!("check_unused_unsafe({:?}, body={:?}, used_unsafe={:?})",
           def_id, body, used_unsafe);

    let mut visitor =  UnusedUnsafeVisitor { used_unsafe, unsafe_blocks };
    hir::intravisit::Visitor::visit_body(&mut visitor, body);
}

fn unsafety_check_result<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId)
                                   -> UnsafetyCheckResult
{
    debug!("unsafety_violations({:?})", def_id);

    // NB: this borrow is valid because all the consumers of
    // `mir_built` force this.
    let mir = &tcx.mir_built(def_id).borrow();

    let visibility_scope_info = match mir.visibility_scope_info {
        ClearOnDecode::Set(ref data) => data,
        ClearOnDecode::Clear => {
            debug!("unsafety_violations: {:?} - remote, skipping", def_id);
            return UnsafetyCheckResult {
                violations: Rc::new([]),
                unsafe_blocks: Rc::new([])
            }
        }
    };

    let param_env = tcx.param_env(def_id);
    let mut checker = UnsafetyChecker::new(
        mir, visibility_scope_info, tcx, param_env);
    checker.visit_mir(mir);

    check_unused_unsafe(tcx, def_id, &checker.used_unsafe, &mut checker.inherited_blocks);
    UnsafetyCheckResult {
        violations: checker.violations.into(),
        unsafe_blocks: checker.inherited_blocks.into()
    }
}

/// Return the NodeId for an enclosing scope that is also `unsafe`
fn is_enclosed(tcx: TyCtxt,
               used_unsafe: &FxHashSet<ast::NodeId>,
               id: ast::NodeId) -> Option<(String, ast::NodeId)> {
    let parent_id = tcx.hir.get_parent_node(id);
    if parent_id != id {
        if used_unsafe.contains(&parent_id) {
            Some(("block".to_string(), parent_id))
        } else if let Some(hir::map::NodeItem(&hir::Item {
            node: hir::ItemFn(_, hir::Unsafety::Unsafe, _, _, _, _),
            ..
        })) = tcx.hir.find(parent_id) {
            Some(("fn".to_string(), parent_id))
        } else {
            is_enclosed(tcx, used_unsafe, parent_id)
        }
    } else {
        None
    }
}

fn report_unused_unsafe(tcx: TyCtxt, used_unsafe: &FxHashSet<ast::NodeId>, id: ast::NodeId) {
    let span = tcx.hir.span(id);
    let mut db = tcx.struct_span_lint_node(UNUSED_UNSAFE, id, span, "unnecessary `unsafe` block");
    db.span_label(span, "unnecessary `unsafe` block");
    if let Some((kind, id)) = is_enclosed(tcx, used_unsafe, id) {
        db.span_note(tcx.hir.span(id),
                     &format!("because it's nested under this `unsafe` {}", kind));
    }
    db.emit();
}

pub fn check_unsafety<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    debug!("check_unsafety({:?})", def_id);

    // closures are handled by their parent fn.
    if tcx.is_closure(def_id) {
        return;
    }

    let UnsafetyCheckResult {
        violations,
        unsafe_blocks
    } = tcx.unsafety_check_result(def_id);

    for &UnsafetyViolation {
        source_info, description, lint_node_id
    } in violations.iter() {
        // Report an error.
        if let Some(lint_node_id) = lint_node_id {
            tcx.lint_node(SAFE_EXTERN_STATICS,
                          lint_node_id,
                          source_info.span,
                          &format!("{} requires unsafe function or \
                                    block (error E0133)", description));
        } else {
            struct_span_err!(
                tcx.sess, source_info.span, E0133,
                "{} requires unsafe function or block", description)
                .span_label(source_info.span, description)
                .emit();
        }
    }

    let mut unsafe_blocks: Vec<_> = unsafe_blocks.into_iter().collect();
    unsafe_blocks.sort();
    let used_unsafe: FxHashSet<_> = unsafe_blocks.iter()
        .flat_map(|&&(id, used)| if used { Some(id) } else { None })
        .collect();
    for &(block_id, is_used) in unsafe_blocks {
        if !is_used {
            report_unused_unsafe(tcx, &used_unsafe, block_id);
        }
    }
}
