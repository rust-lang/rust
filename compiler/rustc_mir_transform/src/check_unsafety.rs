use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::HirId;
use rustc_hir::intravisit;
use rustc_hir::Node;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint::builtin::{UNSAFE_OP_IN_UNSAFE_FN, UNUSED_UNSAFE};
use rustc_session::lint::Level;

use std::ops::Bound;

pub struct UnsafetyChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    body_did: LocalDefId,
    violations: Vec<UnsafetyViolation>,
    source_info: SourceInfo,
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    /// Mark an `unsafe` block as used, so we don't lint it.
    used_unsafe: FxHashSet<hir::HirId>,
    inherited_blocks: Vec<(hir::HirId, bool)>,
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn new(
        body: &'a Body<'tcx>,
        body_did: LocalDefId,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        Self {
            body,
            body_did,
            violations: vec![],
            source_info: SourceInfo::outermost(body.span),
            tcx,
            param_env,
            used_unsafe: Default::default(),
            inherited_blocks: vec![],
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for UnsafetyChecker<'a, 'tcx> {
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        self.source_info = terminator.source_info;
        match terminator.kind {
            TerminatorKind::Goto { .. }
            | TerminatorKind::SwitchInt { .. }
            | TerminatorKind::Drop { .. }
            | TerminatorKind::Yield { .. }
            | TerminatorKind::Assert { .. }
            | TerminatorKind::DropAndReplace { .. }
            | TerminatorKind::GeneratorDrop
            | TerminatorKind::Resume
            | TerminatorKind::Abort
            | TerminatorKind::Return
            | TerminatorKind::Unreachable
            | TerminatorKind::FalseEdge { .. }
            | TerminatorKind::FalseUnwind { .. } => {
                // safe (at least as emitted during MIR construction)
            }

            TerminatorKind::Call { ref func, .. } => {
                let func_ty = func.ty(self.body, self.tcx);
                let sig = func_ty.fn_sig(self.tcx);
                if let hir::Unsafety::Unsafe = sig.unsafety() {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::CallToUnsafeFunction,
                    )
                }

                if let ty::FnDef(func_id, _) = func_ty.kind() {
                    self.check_target_features(*func_id);
                }
            }

            TerminatorKind::InlineAsm { .. } => self.require_unsafe(
                UnsafetyViolationKind::General,
                UnsafetyViolationDetails::UseOfInlineAssembly,
            ),
        }
        self.super_terminator(terminator, location);
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        self.source_info = statement.source_info;
        match statement.kind {
            StatementKind::Assign(..)
            | StatementKind::FakeRead(..)
            | StatementKind::SetDiscriminant { .. }
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Retag { .. }
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::Nop => {
                // safe (at least as emitted during MIR construction)
            }

            StatementKind::LlvmInlineAsm { .. } => self.require_unsafe(
                UnsafetyViolationKind::General,
                UnsafetyViolationDetails::UseOfInlineAssembly,
            ),
            StatementKind::CopyNonOverlapping(..) => unreachable!(),
        }
        self.super_statement(statement, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        match rvalue {
            Rvalue::Aggregate(box ref aggregate, _) => match aggregate {
                &AggregateKind::Array(..) | &AggregateKind::Tuple => {}
                &AggregateKind::Adt(ref def, ..) => {
                    match self.tcx.layout_scalar_valid_range(def.did) {
                        (Bound::Unbounded, Bound::Unbounded) => {}
                        _ => self.require_unsafe(
                            UnsafetyViolationKind::General,
                            UnsafetyViolationDetails::InitializingTypeWith,
                        ),
                    }
                }
                &AggregateKind::Closure(def_id, _) | &AggregateKind::Generator(def_id, _, _) => {
                    let UnsafetyCheckResult { violations, unsafe_blocks } =
                        self.tcx.unsafety_check_result(def_id.expect_local());
                    self.register_violations(&violations, &unsafe_blocks);
                }
            },
            _ => {}
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, _location: Location) {
        // On types with `scalar_valid_range`, prevent
        // * `&mut x.field`
        // * `x.field = y;`
        // * `&x.field` if `field`'s type has interior mutability
        // because either of these would allow modifying the layout constrained field and
        // insert values that violate the layout constraints.
        if context.is_mutating_use() || context.is_borrow() {
            self.check_mut_borrowing_layout_constrained_field(*place, context.is_mutating_use());
        }

        // Some checks below need the extra metainfo of the local declaration.
        let decl = &self.body.local_decls[place.local];

        // Check the base local: it might be an unsafe-to-access static. We only check derefs of the
        // temporary holding the static pointer to avoid duplicate errors
        // <https://github.com/rust-lang/rust/pull/78068#issuecomment-731753506>.
        if decl.internal && place.projection.first() == Some(&ProjectionElem::Deref) {
            // If the projection root is an artifical local that we introduced when
            // desugaring `static`, give a more specific error message
            // (avoid the general "raw pointer" clause below, that would only be confusing).
            if let Some(box LocalInfo::StaticRef { def_id, .. }) = decl.local_info {
                if self.tcx.is_mutable_static(def_id) {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::UseOfMutableStatic,
                    );
                    return;
                } else if self.tcx.is_foreign_item(def_id) {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::UseOfExternStatic,
                    );
                    return;
                }
            }
        }

        // Check for raw pointer `Deref`.
        for (base, proj) in place.iter_projections() {
            if proj == ProjectionElem::Deref {
                let base_ty = base.ty(self.body, self.tcx).ty;
                if base_ty.is_unsafe_ptr() {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::DerefOfRawPointer,
                    )
                }
            }
        }

        // Check for union fields. For this we traverse right-to-left, as the last `Deref` changes
        // whether we *read* the union field or potentially *write* to it (if this place is being assigned to).
        let mut saw_deref = false;
        for (base, proj) in place.iter_projections().rev() {
            if proj == ProjectionElem::Deref {
                saw_deref = true;
                continue;
            }

            let base_ty = base.ty(self.body, self.tcx).ty;
            if base_ty.is_union() {
                // If we did not hit a `Deref` yet and the overall place use is an assignment, the
                // rules are different.
                let assign_to_field = !saw_deref
                    && matches!(
                        context,
                        PlaceContext::MutatingUse(
                            MutatingUseContext::Store
                                | MutatingUseContext::Drop
                                | MutatingUseContext::AsmOutput
                        )
                    );
                // If this is just an assignment, determine if the assigned type needs dropping.
                if assign_to_field {
                    // We have to check the actual type of the assignment, as that determines if the
                    // old value is being dropped.
                    let assigned_ty = place.ty(&self.body.local_decls, self.tcx).ty;
                    // To avoid semver hazard, we only consider `Copy` and `ManuallyDrop` non-dropping.
                    let manually_drop = assigned_ty
                        .ty_adt_def()
                        .map_or(false, |adt_def| adt_def.is_manually_drop());
                    let nodrop = manually_drop
                        || assigned_ty.is_copy_modulo_regions(
                            self.tcx.at(self.source_info.span),
                            self.param_env,
                        );
                    if !nodrop {
                        self.require_unsafe(
                            UnsafetyViolationKind::General,
                            UnsafetyViolationDetails::AssignToDroppingUnionField,
                        );
                    } else {
                        // write to non-drop union field, safe
                    }
                } else {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::AccessToUnionField,
                    )
                }
            }
        }
    }
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn require_unsafe(&mut self, kind: UnsafetyViolationKind, details: UnsafetyViolationDetails) {
        // Violations can turn out to be `UnsafeFn` during analysis, but they should not start out as such.
        assert_ne!(kind, UnsafetyViolationKind::UnsafeFn);

        let source_info = self.source_info;
        let lint_root = self.body.source_scopes[self.source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .lint_root;
        self.register_violations(
            &[UnsafetyViolation { source_info, lint_root, kind, details }],
            &[],
        );
    }

    fn register_violations(
        &mut self,
        violations: &[UnsafetyViolation],
        unsafe_blocks: &[(hir::HirId, bool)],
    ) {
        let safety = self.body.source_scopes[self.source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .safety;
        let within_unsafe = match safety {
            // `unsafe` blocks are required in safe code
            Safety::Safe => {
                for violation in violations {
                    match violation.kind {
                        UnsafetyViolationKind::General => {}
                        UnsafetyViolationKind::UnsafeFn => {
                            bug!("`UnsafetyViolationKind::UnsafeFn` in an `Safe` context")
                        }
                    }
                    if !self.violations.contains(violation) {
                        self.violations.push(*violation)
                    }
                }
                false
            }
            // With the RFC 2585, no longer allow `unsafe` operations in `unsafe fn`s
            Safety::FnUnsafe => {
                for violation in violations {
                    let mut violation = *violation;

                    violation.kind = UnsafetyViolationKind::UnsafeFn;
                    if !self.violations.contains(&violation) {
                        self.violations.push(violation)
                    }
                }
                false
            }
            Safety::BuiltinUnsafe => true,
            Safety::ExplicitUnsafe(hir_id) => {
                // mark unsafe block as used if there are any unsafe operations inside
                if !violations.is_empty() {
                    self.used_unsafe.insert(hir_id);
                }
                true
            }
        };
        self.inherited_blocks.extend(
            unsafe_blocks.iter().map(|&(hir_id, is_used)| (hir_id, is_used && !within_unsafe)),
        );
    }
    fn check_mut_borrowing_layout_constrained_field(
        &mut self,
        place: Place<'tcx>,
        is_mut_use: bool,
    ) {
        for (place_base, elem) in place.iter_projections().rev() {
            match elem {
                // Modifications behind a dereference don't affect the value of
                // the pointer.
                ProjectionElem::Deref => return,
                ProjectionElem::Field(..) => {
                    let ty = place_base.ty(&self.body.local_decls, self.tcx).ty;
                    if let ty::Adt(def, _) = ty.kind() {
                        if self.tcx.layout_scalar_valid_range(def.did)
                            != (Bound::Unbounded, Bound::Unbounded)
                        {
                            let details = if is_mut_use {
                                UnsafetyViolationDetails::MutationOfLayoutConstrainedField

                            // Check `is_freeze` as late as possible to avoid cycle errors
                            // with opaque types.
                            } else if !place
                                .ty(self.body, self.tcx)
                                .ty
                                .is_freeze(self.tcx.at(self.source_info.span), self.param_env)
                            {
                                UnsafetyViolationDetails::BorrowOfLayoutConstrainedField
                            } else {
                                continue;
                            };
                            self.require_unsafe(UnsafetyViolationKind::General, details);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Checks whether calling `func_did` needs an `unsafe` context or not, i.e. whether
    /// the called function has target features the calling function hasn't.
    fn check_target_features(&mut self, func_did: DefId) {
        // Unsafety isn't required on wasm targets. For more information see
        // the corresponding check in typeck/src/collect.rs
        if self.tcx.sess.target.options.is_like_wasm {
            return;
        }

        let callee_features = &self.tcx.codegen_fn_attrs(func_did).target_features;
        let self_features = &self.tcx.codegen_fn_attrs(self.body_did).target_features;

        // Is `callee_features` a subset of `calling_features`?
        if !callee_features.iter().all(|feature| self_features.contains(feature)) {
            self.require_unsafe(
                UnsafetyViolationKind::General,
                UnsafetyViolationDetails::CallToFunctionWith,
            )
        }
    }
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        unsafety_check_result: |tcx, def_id| {
            if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
                tcx.unsafety_check_result_for_const_arg(def)
            } else {
                unsafety_check_result(tcx, ty::WithOptConstParam::unknown(def_id))
            }
        },
        unsafety_check_result_for_const_arg: |tcx, (did, param_did)| {
            unsafety_check_result(
                tcx,
                ty::WithOptConstParam { did, const_param_did: Some(param_did) },
            )
        },
        ..*providers
    };
}

struct UnusedUnsafeVisitor<'a> {
    used_unsafe: &'a FxHashSet<hir::HirId>,
    unsafe_blocks: &'a mut Vec<(hir::HirId, bool)>,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for UnusedUnsafeVisitor<'a> {
    type Map = intravisit::ErasedMap<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }

    fn visit_block(&mut self, block: &'tcx hir::Block<'tcx>) {
        intravisit::walk_block(self, block);

        if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = block.rules {
            self.unsafe_blocks.push((block.hir_id, self.used_unsafe.contains(&block.hir_id)));
        }
    }
}

fn check_unused_unsafe(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    used_unsafe: &FxHashSet<hir::HirId>,
    unsafe_blocks: &mut Vec<(hir::HirId, bool)>,
) {
    let body_id = tcx.hir().maybe_body_owned_by(tcx.hir().local_def_id_to_hir_id(def_id));

    let body_id = match body_id {
        Some(body) => body,
        None => {
            debug!("check_unused_unsafe({:?}) - no body found", def_id);
            return;
        }
    };
    let body = tcx.hir().body(body_id);
    debug!("check_unused_unsafe({:?}, body={:?}, used_unsafe={:?})", def_id, body, used_unsafe);

    let mut visitor = UnusedUnsafeVisitor { used_unsafe, unsafe_blocks };
    intravisit::Visitor::visit_body(&mut visitor, body);
}

fn unsafety_check_result<'tcx>(
    tcx: TyCtxt<'tcx>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &'tcx UnsafetyCheckResult {
    debug!("unsafety_violations({:?})", def);

    // N.B., this borrow is valid because all the consumers of
    // `mir_built` force this.
    let body = &tcx.mir_built(def).borrow();

    let param_env = tcx.param_env(def.did);

    let mut checker = UnsafetyChecker::new(body, def.did, tcx, param_env);
    checker.visit_body(&body);

    check_unused_unsafe(tcx, def.did, &checker.used_unsafe, &mut checker.inherited_blocks);

    tcx.arena.alloc(UnsafetyCheckResult {
        violations: checker.violations.into(),
        unsafe_blocks: checker.inherited_blocks.into(),
    })
}

/// Returns the `HirId` for an enclosing scope that is also `unsafe`.
fn is_enclosed(
    tcx: TyCtxt<'_>,
    used_unsafe: &FxHashSet<hir::HirId>,
    id: hir::HirId,
    unsafe_op_in_unsafe_fn_allowed: bool,
) -> Option<(&'static str, hir::HirId)> {
    let parent_id = tcx.hir().get_parent_node(id);
    if parent_id != id {
        if used_unsafe.contains(&parent_id) {
            Some(("block", parent_id))
        } else if let Some(Node::Item(&hir::Item {
            kind: hir::ItemKind::Fn(ref sig, _, _), ..
        })) = tcx.hir().find(parent_id)
        {
            if sig.header.unsafety == hir::Unsafety::Unsafe && unsafe_op_in_unsafe_fn_allowed {
                Some(("fn", parent_id))
            } else {
                None
            }
        } else {
            is_enclosed(tcx, used_unsafe, parent_id, unsafe_op_in_unsafe_fn_allowed)
        }
    } else {
        None
    }
}

fn report_unused_unsafe(tcx: TyCtxt<'_>, used_unsafe: &FxHashSet<hir::HirId>, id: hir::HirId) {
    let span = tcx.sess.source_map().guess_head_span(tcx.hir().span(id));
    tcx.struct_span_lint_hir(UNUSED_UNSAFE, id, span, |lint| {
        let msg = "unnecessary `unsafe` block";
        let mut db = lint.build(msg);
        db.span_label(span, msg);
        if let Some((kind, id)) =
            is_enclosed(tcx, used_unsafe, id, unsafe_op_in_unsafe_fn_allowed(tcx, id))
        {
            db.span_label(
                tcx.sess.source_map().guess_head_span(tcx.hir().span(id)),
                format!("because it's nested under this `unsafe` {}", kind),
            );
        }
        db.emit();
    });
}

pub fn check_unsafety(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    debug!("check_unsafety({:?})", def_id);

    // closures are handled by their parent fn.
    if tcx.is_closure(def_id.to_def_id()) {
        return;
    }

    let UnsafetyCheckResult { violations, unsafe_blocks } = tcx.unsafety_check_result(def_id);

    for &UnsafetyViolation { source_info, lint_root, kind, details } in violations.iter() {
        let (description, note) = details.description_and_note();

        // Report an error.
        let unsafe_fn_msg =
            if unsafe_op_in_unsafe_fn_allowed(tcx, lint_root) { " function or" } else { "" };

        match kind {
            UnsafetyViolationKind::General => {
                // once
                struct_span_err!(
                    tcx.sess,
                    source_info.span,
                    E0133,
                    "{} is unsafe and requires unsafe{} block",
                    description,
                    unsafe_fn_msg,
                )
                .span_label(source_info.span, description)
                .note(note)
                .emit();
            }
            UnsafetyViolationKind::UnsafeFn => tcx.struct_span_lint_hir(
                UNSAFE_OP_IN_UNSAFE_FN,
                lint_root,
                source_info.span,
                |lint| {
                    lint.build(&format!(
                        "{} is unsafe and requires unsafe block (error E0133)",
                        description,
                    ))
                    .span_label(source_info.span, description)
                    .note(note)
                    .emit();
                },
            ),
        }
    }

    let (mut unsafe_used, mut unsafe_unused): (FxHashSet<_>, Vec<_>) = Default::default();
    for &(block_id, is_used) in unsafe_blocks.iter() {
        if is_used {
            unsafe_used.insert(block_id);
        } else {
            unsafe_unused.push(block_id);
        }
    }
    // The unused unsafe blocks might not be in source order; sort them so that the unused unsafe
    // error messages are properly aligned and the issue-45107 and lint-unused-unsafe tests pass.
    unsafe_unused.sort_by_cached_key(|hir_id| tcx.hir().span(*hir_id));

    for &block_id in &unsafe_unused {
        report_unused_unsafe(tcx, &unsafe_used, block_id);
    }
}

fn unsafe_op_in_unsafe_fn_allowed(tcx: TyCtxt<'_>, id: HirId) -> bool {
    tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, id).0 == Level::Allow
}
