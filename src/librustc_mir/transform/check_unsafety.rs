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
use rustc_data_structures::sync::Lrc;

use rustc::ty::query::Providers;
use rustc::ty::{self, TyCtxt};
use rustc::hir;
use rustc::hir::Node;
use rustc::hir::def_id::DefId;
use rustc::lint::builtin::{SAFE_EXTERN_STATICS, SAFE_PACKED_BORROWS, UNUSED_UNSAFE};
use rustc::mir::*;
use rustc::mir::visit::{PlaceContext, Visitor, MutatingUseContext};

use syntax::ast;
use syntax::symbol::Symbol;

use util;

pub struct UnsafetyChecker<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    min_const_fn: bool,
    source_scope_local_data: &'a IndexVec<SourceScope, SourceScopeLocalData>,
    violations: Vec<UnsafetyViolation>,
    source_info: SourceInfo,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    used_unsafe: FxHashSet<ast::NodeId>,
    inherited_blocks: Vec<(ast::NodeId, bool)>,
}

impl<'a, 'gcx, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn new(
        min_const_fn: bool,
        mir: &'a Mir<'tcx>,
        source_scope_local_data: &'a IndexVec<SourceScope, SourceScopeLocalData>,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        Self {
            mir,
            min_const_fn,
            source_scope_local_data,
            violations: vec![],
            source_info: SourceInfo {
                span: mir.span,
                scope: OUTERMOST_SOURCE_SCOPE
            },
            tcx,
            param_env,
            used_unsafe: Default::default(),
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
            TerminatorKind::Abort |
            TerminatorKind::Return |
            TerminatorKind::Unreachable |
            TerminatorKind::FalseEdges { .. } |
            TerminatorKind::FalseUnwind { .. } => {
                // safe (at least as emitted during MIR construction)
            }

            TerminatorKind::Call { ref func, .. } => {
                let func_ty = func.ty(self.mir, self.tcx);
                let sig = func_ty.fn_sig(self.tcx);
                if let hir::Unsafety::Unsafe = sig.unsafety() {
                    self.require_unsafe("call to unsafe function",
                        "consult the function's documentation for information on how to avoid \
                         undefined behavior")
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
            StatementKind::FakeRead(..) |
            StatementKind::SetDiscriminant { .. } |
            StatementKind::StorageLive(..) |
            StatementKind::StorageDead(..) |
            StatementKind::Retag { .. } |
            StatementKind::EscapeToRaw { .. } |
            StatementKind::AscribeUserType(..) |
            StatementKind::Nop => {
                // safe (at least as emitted during MIR construction)
            }

            StatementKind::InlineAsm { .. } => {
                self.require_unsafe("use of inline assembly",
                    "inline assembly is entirely unchecked and can cause undefined behavior")
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
                    self.register_violations(&violations, &unsafe_blocks);
                }
            }
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(&mut self,
                    place: &Place<'tcx>,
                    context: PlaceContext<'tcx>,
                    location: Location) {
        if context.is_borrow() {
            if util::is_disaligned(self.tcx, self.mir, self.param_env, place) {
                let source_info = self.source_info;
                let lint_root =
                    self.source_scope_local_data[source_info.scope].lint_root;
                self.register_violations(&[UnsafetyViolation {
                    source_info,
                    description: Symbol::intern("borrow of packed field").as_interned_str(),
                    details:
                        Symbol::intern("fields of packed structs might be misaligned: \
                                        dereferencing a misaligned pointer or even just creating a \
                                        misaligned reference is undefined behavior")
                            .as_interned_str(),
                    kind: UnsafetyViolationKind::BorrowPacked(lint_root)
                }], &[]);
            }
        }

        match place {
            &Place::Projection(box Projection {
                ref base, ref elem
            }) => {
                let old_source_info = self.source_info;
                if let &Place::Local(local) = base {
                    if self.mir.local_decls[local].internal {
                        // Internal locals are used in the `move_val_init` desugaring.
                        // We want to check unsafety against the source info of the
                        // desugaring, rather than the source info of the RHS.
                        self.source_info = self.mir.local_decls[local].source_info;
                    }
                }
                let base_ty = base.ty(self.mir, self.tcx).to_ty(self.tcx);
                match base_ty.sty {
                    ty::RawPtr(..) => {
                        self.require_unsafe("dereference of raw pointer",
                            "raw pointers may be NULL, dangling or unaligned; they can violate \
                             aliasing rules and cause data races: all of these are undefined \
                             behavior")
                    }
                    ty::Adt(adt, _) => {
                        if adt.is_union() {
                            if context == PlaceContext::MutatingUse(MutatingUseContext::Store) ||
                                context == PlaceContext::MutatingUse(MutatingUseContext::Drop) ||
                                context == PlaceContext::MutatingUse(
                                    MutatingUseContext::AsmOutput
                                )
                            {
                                let elem_ty = match elem {
                                    &ProjectionElem::Field(_, ty) => ty,
                                    _ => span_bug!(
                                        self.source_info.span,
                                        "non-field projection {:?} from union?",
                                        place)
                                };
                                if elem_ty.moves_by_default(self.tcx, self.param_env,
                                                            self.source_info.span) {
                                    self.require_unsafe(
                                        "assignment to non-`Copy` union field",
                                        "the previous content of the field will be dropped, which \
                                         causes undefined behavior if the field was not properly \
                                         initialized")
                                } else {
                                    // write to non-move union, safe
                                }
                            } else {
                                self.require_unsafe("access to union field",
                                    "the field may not be properly initialized: using \
                                     uninitialized data will cause undefined behavior")
                            }
                        }
                    }
                    _ => {}
                }
                self.source_info = old_source_info;
            }
            &Place::Local(..) => {
                // locals are safe
            }
            &Place::Promoted(_) => {
                bug!("unsafety checking should happen before promotion")
            }
            &Place::Static(box Static { def_id, ty: _ }) => {
                if self.tcx.is_static(def_id) == Some(hir::Mutability::MutMutable) {
                    self.require_unsafe("use of mutable static",
                        "mutable statics can be mutated by multiple threads: aliasing violations \
                         or data races will cause undefined behavior");
                } else if self.tcx.is_foreign_item(def_id) {
                    let source_info = self.source_info;
                    let lint_root =
                        self.source_scope_local_data[source_info.scope].lint_root;
                    self.register_violations(&[UnsafetyViolation {
                        source_info,
                        description: Symbol::intern("use of extern static").as_interned_str(),
                        details:
                            Symbol::intern("extern statics are not controlled by the Rust type \
                                            system: invalid data, aliasing violations or data \
                                            races will cause undefined behavior")
                                .as_interned_str(),
                        kind: UnsafetyViolationKind::ExternStatic(lint_root)
                    }], &[]);
                }
            }
        };
        self.super_place(place, context, location);
    }
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn require_unsafe(&mut self,
                      description: &'static str,
                      details: &'static str)
    {
        let source_info = self.source_info;
        self.register_violations(&[UnsafetyViolation {
            source_info,
            description: Symbol::intern(description).as_interned_str(),
            details: Symbol::intern(details).as_interned_str(),
            kind: UnsafetyViolationKind::General,
        }], &[]);
    }

    fn register_violations(&mut self,
                           violations: &[UnsafetyViolation],
                           unsafe_blocks: &[(ast::NodeId, bool)]) {
        if self.min_const_fn {
            for violation in violations {
                let mut violation = violation.clone();
                violation.kind = UnsafetyViolationKind::MinConstFn;
                if !self.violations.contains(&violation) {
                    self.violations.push(violation)
                }
            }
        }
        let within_unsafe = match self.source_scope_local_data[self.source_info.scope].safety {
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
        unsafe_derive_on_repr_packed,
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

    let source_scope_local_data = match mir.source_scope_local_data {
        ClearCrossCrate::Set(ref data) => data,
        ClearCrossCrate::Clear => {
            debug!("unsafety_violations: {:?} - remote, skipping", def_id);
            return UnsafetyCheckResult {
                violations: Lrc::new([]),
                unsafe_blocks: Lrc::new([])
            }
        }
    };

    let param_env = tcx.param_env(def_id);
    let mut checker = UnsafetyChecker::new(
        tcx.is_const_fn(def_id) && tcx.is_min_const_fn(def_id),
        mir, source_scope_local_data, tcx, param_env);
    checker.visit_mir(mir);

    check_unused_unsafe(tcx, def_id, &checker.used_unsafe, &mut checker.inherited_blocks);
    UnsafetyCheckResult {
        violations: checker.violations.into(),
        unsafe_blocks: checker.inherited_blocks.into()
    }
}

fn unsafe_derive_on_repr_packed<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) {
    let lint_node_id = match tcx.hir.as_local_node_id(def_id) {
        Some(node_id) => node_id,
        None => bug!("checking unsafety for non-local def id {:?}", def_id)
    };

    // FIXME: when we make this a hard error, this should have its
    // own error code.
    let message = if tcx.generics_of(def_id).own_counts().types != 0 {
        "#[derive] can't be used on a #[repr(packed)] struct with \
         type parameters (error E0133)".to_string()
    } else {
        "#[derive] can't be used on a #[repr(packed)] struct that \
         does not derive Copy (error E0133)".to_string()
    };
    tcx.lint_node(SAFE_PACKED_BORROWS,
                  lint_node_id,
                  tcx.def_span(def_id),
                  &message);
}

/// Return the NodeId for an enclosing scope that is also `unsafe`
fn is_enclosed(tcx: TyCtxt,
               used_unsafe: &FxHashSet<ast::NodeId>,
               id: ast::NodeId) -> Option<(String, ast::NodeId)> {
    let parent_id = tcx.hir.get_parent_node(id);
    if parent_id != id {
        if used_unsafe.contains(&parent_id) {
            Some(("block".to_string(), parent_id))
        } else if let Some(Node::Item(&hir::Item {
            node: hir::ItemKind::Fn(_, header, _, _),
            ..
        })) = tcx.hir.find(parent_id) {
            match header.unsafety {
                hir::Unsafety::Unsafe => Some(("fn".to_string(), parent_id)),
                hir::Unsafety::Normal => None,
            }
        } else {
            is_enclosed(tcx, used_unsafe, parent_id)
        }
    } else {
        None
    }
}

fn report_unused_unsafe(tcx: TyCtxt, used_unsafe: &FxHashSet<ast::NodeId>, id: ast::NodeId) {
    let span = tcx.sess.source_map().def_span(tcx.hir.span(id));
    let msg = "unnecessary `unsafe` block";
    let mut db = tcx.struct_span_lint_node(UNUSED_UNSAFE, id, span, msg);
    db.span_label(span, msg);
    if let Some((kind, id)) = is_enclosed(tcx, used_unsafe, id) {
        db.span_label(tcx.sess.source_map().def_span(tcx.hir.span(id)),
                      format!("because it's nested under this `unsafe` {}", kind));
    }
    db.emit();
}

fn builtin_derive_def_id<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>, def_id: DefId) -> Option<DefId> {
    debug!("builtin_derive_def_id({:?})", def_id);
    if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
        if tcx.has_attr(impl_def_id, "automatically_derived") {
            debug!("builtin_derive_def_id({:?}) - is {:?}", def_id, impl_def_id);
            Some(impl_def_id)
        } else {
            debug!("builtin_derive_def_id({:?}) - not automatically derived", def_id);
            None
        }
    } else {
        debug!("builtin_derive_def_id({:?}) - not a method", def_id);
        None
    }
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
        source_info, description, details, kind
    } in violations.iter() {
        // Report an error.
        match kind {
            UnsafetyViolationKind::General => {
                struct_span_err!(
                    tcx.sess, source_info.span, E0133,
                    "{} is unsafe and requires unsafe function or block", description)
                    .span_label(source_info.span, &description.as_str()[..])
                    .note(&details.as_str()[..])
                    .emit();
            }
            UnsafetyViolationKind::MinConstFn => {
                tcx.sess.struct_span_err(
                    source_info.span,
                    &format!("{} is unsafe and unsafe operations \
                            are not allowed in const fn", description))
                    .span_label(source_info.span, &description.as_str()[..])
                    .note(&details.as_str()[..])
                    .emit();
            }
            UnsafetyViolationKind::ExternStatic(lint_node_id) => {
                tcx.lint_node_note(SAFE_EXTERN_STATICS,
                              lint_node_id,
                              source_info.span,
                              &format!("{} is unsafe and requires unsafe function or block \
                                        (error E0133)", &description.as_str()[..]),
                              &details.as_str()[..]);
            }
            UnsafetyViolationKind::BorrowPacked(lint_node_id) => {
                if let Some(impl_def_id) = builtin_derive_def_id(tcx, def_id) {
                    tcx.unsafe_derive_on_repr_packed(impl_def_id);
                } else {
                    tcx.lint_node_note(SAFE_PACKED_BORROWS,
                                  lint_node_id,
                                  source_info.span,
                                  &format!("{} is unsafe and requires unsafe function or block \
                                            (error E0133)", &description.as_str()[..]),
                                  &details.as_str()[..]);
                }
            }
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
