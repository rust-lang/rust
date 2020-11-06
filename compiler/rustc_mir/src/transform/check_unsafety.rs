use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::HirId;
use rustc_hir::intravisit;
use rustc_hir::Node;
use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::*;
use rustc_middle::ty::cast::CastTy;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint::builtin::{SAFE_PACKED_BORROWS, UNSAFE_OP_IN_UNSAFE_FN, UNUSED_UNSAFE};
use rustc_session::lint::Level;
use rustc_span::symbol::sym;

use std::ops::Bound;

use crate::const_eval::is_min_const_fn;
use crate::util;

pub struct UnsafetyChecker<'a, 'tcx> {
    body: &'a Body<'tcx>,
    body_did: LocalDefId,
    const_context: bool,
    min_const_fn: bool,
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
        const_context: bool,
        min_const_fn: bool,
        body: &'a Body<'tcx>,
        body_did: LocalDefId,
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Self {
        // sanity check
        if min_const_fn {
            assert!(const_context);
        }
        Self {
            body,
            body_did,
            const_context,
            min_const_fn,
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
                        UnsafetyViolationKind::GeneralAndConstFn,
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
                            UnsafetyViolationKind::GeneralAndConstFn,
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
            // casting pointers to ints is unsafe in const fn because the const evaluator cannot
            // possibly know what the result of various operations like `address / 2` would be
            // pointers during const evaluation have no integral address, only an abstract one
            Rvalue::Cast(CastKind::Misc, ref operand, cast_ty)
                if self.const_context && self.tcx.features().const_raw_ptr_to_usize_cast =>
            {
                let operand_ty = operand.ty(self.body, self.tcx);
                let cast_in = CastTy::from_ty(operand_ty).expect("bad input type for cast");
                let cast_out = CastTy::from_ty(cast_ty).expect("bad output type for cast");
                match (cast_in, cast_out) {
                    (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) => {
                        self.require_unsafe(
                            UnsafetyViolationKind::General,
                            UnsafetyViolationDetails::CastOfPointerToInt,
                        );
                    }
                    _ => {}
                }
            }
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

        if context.is_borrow() {
            if util::is_disaligned(self.tcx, self.body, self.param_env, *place) {
                self.require_unsafe(
                    UnsafetyViolationKind::BorrowPacked,
                    UnsafetyViolationDetails::BorrowOfPackedField,
                );
            }
        }

        for (i, elem) in place.projection.iter().enumerate() {
            let proj_base = &place.projection[..i];
            if context.is_borrow() {
                if util::is_disaligned(self.tcx, self.body, self.param_env, *place) {
                    self.require_unsafe(
                        UnsafetyViolationKind::BorrowPacked,
                        UnsafetyViolationDetails::BorrowOfPackedField,
                    );
                }
            }
            let source_info = self.source_info;
            if let [] = proj_base {
                let decl = &self.body.local_decls[place.local];
                if decl.internal {
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
                    } else {
                        // Internal locals are used in the `move_val_init` desugaring.
                        // We want to check unsafety against the source info of the
                        // desugaring, rather than the source info of the RHS.
                        self.source_info = self.body.local_decls[place.local].source_info;
                    }
                }
            }
            let base_ty = Place::ty_from(place.local, proj_base, self.body, self.tcx).ty;
            match base_ty.kind() {
                ty::RawPtr(..) => self.require_unsafe(
                    UnsafetyViolationKind::GeneralAndConstFn,
                    UnsafetyViolationDetails::DerefOfRawPointer,
                ),
                ty::Adt(adt, _) => {
                    if adt.is_union() {
                        if context == PlaceContext::MutatingUse(MutatingUseContext::Store)
                            || context == PlaceContext::MutatingUse(MutatingUseContext::Drop)
                            || context == PlaceContext::MutatingUse(MutatingUseContext::AsmOutput)
                        {
                            let elem_ty = match elem {
                                ProjectionElem::Field(_, ty) => ty,
                                _ => span_bug!(
                                    self.source_info.span,
                                    "non-field projection {:?} from union?",
                                    place
                                ),
                            };
                            if !elem_ty.is_copy_modulo_regions(
                                self.tcx.at(self.source_info.span),
                                self.param_env,
                            ) {
                                self.require_unsafe(
                                    UnsafetyViolationKind::GeneralAndConstFn,
                                    UnsafetyViolationDetails::AssignToNonCopyUnionField,
                                )
                            } else {
                                // write to non-move union, safe
                            }
                        } else {
                            self.require_unsafe(
                                UnsafetyViolationKind::GeneralAndConstFn,
                                UnsafetyViolationDetails::AccessToUnionField,
                            )
                        }
                    }
                }
                _ => {}
            }
            self.source_info = source_info;
        }
    }
}

impl<'a, 'tcx> UnsafetyChecker<'a, 'tcx> {
    fn require_unsafe(&mut self, kind: UnsafetyViolationKind, details: UnsafetyViolationDetails) {
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
                    let mut violation = *violation;
                    match violation.kind {
                        UnsafetyViolationKind::GeneralAndConstFn
                        | UnsafetyViolationKind::General => {}
                        UnsafetyViolationKind::BorrowPacked => {
                            if self.min_const_fn {
                                // const fns don't need to be backwards compatible and can
                                // emit these violations as a hard error instead of a backwards
                                // compat lint
                                violation.kind = UnsafetyViolationKind::General;
                            }
                        }
                        UnsafetyViolationKind::UnsafeFn
                        | UnsafetyViolationKind::UnsafeFnBorrowPacked => {
                            bug!("`UnsafetyViolationKind::UnsafeFn` in an `Safe` context")
                        }
                    }
                    if !self.violations.contains(&violation) {
                        self.violations.push(violation)
                    }
                }
                false
            }
            // With the RFC 2585, no longer allow `unsafe` operations in `unsafe fn`s
            Safety::FnUnsafe if self.tcx.features().unsafe_block_in_unsafe_fn => {
                for violation in violations {
                    let mut violation = *violation;

                    if violation.kind == UnsafetyViolationKind::BorrowPacked {
                        violation.kind = UnsafetyViolationKind::UnsafeFnBorrowPacked;
                    } else {
                        violation.kind = UnsafetyViolationKind::UnsafeFn;
                    }
                    if !self.violations.contains(&violation) {
                        self.violations.push(violation)
                    }
                }
                false
            }
            // `unsafe` function bodies allow unsafe without additional unsafe blocks (before RFC 2585)
            Safety::BuiltinUnsafe | Safety::FnUnsafe => true,
            Safety::ExplicitUnsafe(hir_id) => {
                // mark unsafe block as used if there are any unsafe operations inside
                if !violations.is_empty() {
                    self.used_unsafe.insert(hir_id);
                }
                // only some unsafety is allowed in const fn
                if self.min_const_fn {
                    for violation in violations {
                        match violation.kind {
                            // these unsafe things are stable in const fn
                            UnsafetyViolationKind::GeneralAndConstFn => {}
                            // these things are forbidden in const fns
                            UnsafetyViolationKind::General
                            | UnsafetyViolationKind::BorrowPacked => {
                                let mut violation = *violation;
                                // const fns don't need to be backwards compatible and can
                                // emit these violations as a hard error instead of a backwards
                                // compat lint
                                violation.kind = UnsafetyViolationKind::General;
                                if !self.violations.contains(&violation) {
                                    self.violations.push(violation)
                                }
                            }
                            UnsafetyViolationKind::UnsafeFn
                            | UnsafetyViolationKind::UnsafeFnBorrowPacked => bug!(
                                "`UnsafetyViolationKind::UnsafeFn` in an `ExplicitUnsafe` context"
                            ),
                        }
                    }
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
        let mut cursor = place.projection.as_ref();
        while let &[ref proj_base @ .., elem] = cursor {
            cursor = proj_base;

            match elem {
                // Modifications behind a dereference don't affect the value of
                // the pointer.
                ProjectionElem::Deref => return,
                ProjectionElem::Field(..) => {
                    let ty =
                        Place::ty_from(place.local, proj_base, &self.body.local_decls, self.tcx).ty;
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
                            self.require_unsafe(UnsafetyViolationKind::GeneralAndConstFn, details);
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
        let callee_features = &self.tcx.codegen_fn_attrs(func_did).target_features;
        let self_features = &self.tcx.codegen_fn_attrs(self.body_did).target_features;

        // Is `callee_features` a subset of `calling_features`?
        if !callee_features.iter().all(|feature| self_features.contains(feature)) {
            self.require_unsafe(
                UnsafetyViolationKind::GeneralAndConstFn,
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
        unsafe_derive_on_repr_packed,
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

    let id = tcx.hir().local_def_id_to_hir_id(def.did);
    let (const_context, min_const_fn) = match tcx.hir().body_owner_kind(id) {
        hir::BodyOwnerKind::Closure => (false, false),
        hir::BodyOwnerKind::Fn => {
            (tcx.is_const_fn_raw(def.did.to_def_id()), is_min_const_fn(tcx, def.did.to_def_id()))
        }
        hir::BodyOwnerKind::Const | hir::BodyOwnerKind::Static(_) => (true, false),
    };
    let mut checker =
        UnsafetyChecker::new(const_context, min_const_fn, body, def.did, tcx, param_env);
    checker.visit_body(&body);

    check_unused_unsafe(tcx, def.did, &checker.used_unsafe, &mut checker.inherited_blocks);

    tcx.arena.alloc(UnsafetyCheckResult {
        violations: checker.violations.into(),
        unsafe_blocks: checker.inherited_blocks.into(),
    })
}

fn unsafe_derive_on_repr_packed(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    let lint_hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    tcx.struct_span_lint_hir(SAFE_PACKED_BORROWS, lint_hir_id, tcx.def_span(def_id), |lint| {
        // FIXME: when we make this a hard error, this should have its
        // own error code.
        let message = if tcx.generics_of(def_id).own_requires_monomorphization() {
            "`#[derive]` can't be used on a `#[repr(packed)]` struct with \
             type or const parameters (error E0133)"
                .to_string()
        } else {
            "`#[derive]` can't be used on a `#[repr(packed)]` struct that \
             does not derive Copy (error E0133)"
                .to_string()
        };
        lint.build(&message).emit()
    });
}

/// Returns the `HirId` for an enclosing scope that is also `unsafe`.
fn is_enclosed(
    tcx: TyCtxt<'_>,
    used_unsafe: &FxHashSet<hir::HirId>,
    id: hir::HirId,
) -> Option<(String, hir::HirId)> {
    let parent_id = tcx.hir().get_parent_node(id);
    if parent_id != id {
        if used_unsafe.contains(&parent_id) {
            Some(("block".to_string(), parent_id))
        } else if let Some(Node::Item(&hir::Item {
            kind: hir::ItemKind::Fn(ref sig, _, _), ..
        })) = tcx.hir().find(parent_id)
        {
            if sig.header.unsafety == hir::Unsafety::Unsafe
                && !tcx.features().unsafe_block_in_unsafe_fn
            {
                Some(("fn".to_string(), parent_id))
            } else {
                None
            }
        } else {
            is_enclosed(tcx, used_unsafe, parent_id)
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
        if let Some((kind, id)) = is_enclosed(tcx, used_unsafe, id) {
            db.span_label(
                tcx.sess.source_map().guess_head_span(tcx.hir().span(id)),
                format!("because it's nested under this `unsafe` {}", kind),
            );
        }
        db.emit();
    });
}

fn builtin_derive_def_id(tcx: TyCtxt<'_>, def_id: DefId) -> Option<DefId> {
    debug!("builtin_derive_def_id({:?})", def_id);
    if let Some(impl_def_id) = tcx.impl_of_method(def_id) {
        if tcx.has_attr(impl_def_id, sym::automatically_derived) {
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
            UnsafetyViolationKind::GeneralAndConstFn | UnsafetyViolationKind::General => {
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
            UnsafetyViolationKind::BorrowPacked => {
                if let Some(impl_def_id) = builtin_derive_def_id(tcx, def_id.to_def_id()) {
                    // If a method is defined in the local crate,
                    // the impl containing that method should also be.
                    tcx.ensure().unsafe_derive_on_repr_packed(impl_def_id.expect_local());
                } else {
                    tcx.struct_span_lint_hir(
                        SAFE_PACKED_BORROWS,
                        lint_root,
                        source_info.span,
                        |lint| {
                            lint.build(&format!(
                                "{} is unsafe and requires unsafe{} block (error E0133)",
                                description, unsafe_fn_msg,
                            ))
                            .note(note)
                            .emit()
                        },
                    )
                }
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
            UnsafetyViolationKind::UnsafeFnBorrowPacked => {
                // When `unsafe_op_in_unsafe_fn` is disallowed, the behavior of safe and unsafe functions
                // should be the same in terms of warnings and errors. Therefore, with `#[warn(safe_packed_borrows)]`,
                // a safe packed borrow should emit a warning *but not an error* in an unsafe function,
                // just like in a safe function, even if `unsafe_op_in_unsafe_fn` is `deny`.
                //
                // Also, `#[warn(unsafe_op_in_unsafe_fn)]` can't cause any new errors. Therefore, with
                // `#[deny(safe_packed_borrows)]` and `#[warn(unsafe_op_in_unsafe_fn)]`, a packed borrow
                // should only issue a warning for the sake of backwards compatibility.
                //
                // The solution those 2 expectations is to always take the minimum of both lints.
                // This prevent any new errors (unless both lints are explicitly set to `deny`).
                let lint = if tcx.lint_level_at_node(SAFE_PACKED_BORROWS, lint_root).0
                    <= tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, lint_root).0
                {
                    SAFE_PACKED_BORROWS
                } else {
                    UNSAFE_OP_IN_UNSAFE_FN
                };
                tcx.struct_span_lint_hir(&lint, lint_root, source_info.span, |lint| {
                    lint.build(&format!(
                        "{} is unsafe and requires unsafe block (error E0133)",
                        description,
                    ))
                    .span_label(source_info.span, description)
                    .note(note)
                    .emit();
                })
            }
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
