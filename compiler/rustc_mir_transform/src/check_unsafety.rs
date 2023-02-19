use rustc_data_structures::fx::FxHashSet;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::hir_id::HirId;
use rustc_hir::intravisit;
use rustc_hir::{BlockCheckMode, ExprKind, Node};
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

    /// Used `unsafe` blocks in this function. This is used for the "unused_unsafe" lint.
    used_unsafe_blocks: FxHashSet<HirId>,
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
            used_unsafe_blocks: Default::default(),
        }
    }
}

impl<'tcx> Visitor<'tcx> for UnsafetyChecker<'_, 'tcx> {
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
                let func_id =
                    if let ty::FnDef(func_id, _) = func_ty.kind() { Some(func_id) } else { None };
                let sig = func_ty.fn_sig(self.tcx);
                if let hir::Unsafety::Unsafe = sig.unsafety() {
                    self.require_unsafe(
                        UnsafetyViolationKind::General,
                        UnsafetyViolationDetails::CallToUnsafeFunction,
                    )
                }

                if let Some(func_id) = func_id {
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
            | StatementKind::Deinit(..)
            | StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Retag { .. }
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(..)
            | StatementKind::Intrinsic(..)
            | StatementKind::ConstEvalCounter
            | StatementKind::Nop => {
                // safe (at least as emitted during MIR construction)
            }
        }
        self.super_statement(statement, location);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        match rvalue {
            Rvalue::Aggregate(box ref aggregate, _) => match aggregate {
                &AggregateKind::Array(..) | &AggregateKind::Tuple => {}
                &AggregateKind::Adt(adt_did, ..) => {
                    match self.tcx.layout_scalar_valid_range(adt_did) {
                        (Bound::Unbounded, Bound::Unbounded) => {}
                        _ => self.require_unsafe(
                            UnsafetyViolationKind::General,
                            UnsafetyViolationDetails::InitializingTypeWith,
                        ),
                    }
                }
                &AggregateKind::Closure(def_id, _) | &AggregateKind::Generator(def_id, _, _) => {
                    let def_id = def_id.expect_local();
                    let UnsafetyCheckResult { violations, used_unsafe_blocks, .. } =
                        self.tcx.unsafety_check_result(def_id);
                    self.register_violations(violations, used_unsafe_blocks.iter().copied());
                }
            },
            _ => {}
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_operand(&mut self, op: &Operand<'tcx>, location: Location) {
        if let Operand::Constant(constant) = op {
            let maybe_uneval = match constant.literal {
                ConstantKind::Val(..) | ConstantKind::Ty(_) => None,
                ConstantKind::Unevaluated(uv, _) => Some(uv),
            };

            if let Some(uv) = maybe_uneval {
                if uv.promoted.is_none() {
                    let def_id = uv.def.def_id_for_type_of();
                    if self.tcx.def_kind(def_id) == DefKind::InlineConst {
                        let local_def_id = def_id.expect_local();
                        let UnsafetyCheckResult { violations, used_unsafe_blocks, .. } =
                            self.tcx.unsafety_check_result(local_def_id);
                        self.register_violations(violations, used_unsafe_blocks.iter().copied());
                    }
                }
            }
        }
        self.super_operand(op, location);
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

        // Some checks below need the extra meta info of the local declaration.
        let decl = &self.body.local_decls[place.local];

        // Check the base local: it might be an unsafe-to-access static. We only check derefs of the
        // temporary holding the static pointer to avoid duplicate errors
        // <https://github.com/rust-lang/rust/pull/78068#issuecomment-731753506>.
        if decl.internal && place.projection.first() == Some(&ProjectionElem::Deref) {
            // If the projection root is an artificial local that we introduced when
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
                    if assigned_ty.needs_drop(self.tcx, self.param_env) {
                        // This would be unsafe, but should be outright impossible since we reject such unions.
                        self.tcx.sess.delay_span_bug(
                            self.source_info.span,
                            format!("union fields that need dropping should be impossible: {assigned_ty}")
                        );
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

impl<'tcx> UnsafetyChecker<'_, 'tcx> {
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
            [&UnsafetyViolation { source_info, lint_root, kind, details }],
            [],
        );
    }

    fn register_violations<'a>(
        &mut self,
        violations: impl IntoIterator<Item = &'a UnsafetyViolation>,
        new_used_unsafe_blocks: impl IntoIterator<Item = HirId>,
    ) {
        let safety = self.body.source_scopes[self.source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .safety;
        match safety {
            // `unsafe` blocks are required in safe code
            Safety::Safe => violations.into_iter().for_each(|&violation| {
                match violation.kind {
                    UnsafetyViolationKind::General => {}
                    UnsafetyViolationKind::UnsafeFn => {
                        bug!("`UnsafetyViolationKind::UnsafeFn` in an `Safe` context")
                    }
                }
                if !self.violations.contains(&violation) {
                    self.violations.push(violation)
                }
            }),
            // With the RFC 2585, no longer allow `unsafe` operations in `unsafe fn`s
            Safety::FnUnsafe => violations.into_iter().for_each(|&(mut violation)| {
                violation.kind = UnsafetyViolationKind::UnsafeFn;
                if !self.violations.contains(&violation) {
                    self.violations.push(violation)
                }
            }),
            Safety::BuiltinUnsafe => {}
            Safety::ExplicitUnsafe(hir_id) => violations.into_iter().for_each(|_violation| {
                self.used_unsafe_blocks.insert(hir_id);
            }),
        };

        new_used_unsafe_blocks.into_iter().for_each(|hir_id| {
            self.used_unsafe_blocks.insert(hir_id);
        });
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
                        if self.tcx.layout_scalar_valid_range(def.did())
                            != (Bound::Unbounded, Bound::Unbounded)
                        {
                            let details = if is_mut_use {
                                UnsafetyViolationDetails::MutationOfLayoutConstrainedField

                            // Check `is_freeze` as late as possible to avoid cycle errors
                            // with opaque types.
                            } else if !place
                                .ty(self.body, self.tcx)
                                .ty
                                .is_freeze(self.tcx, self.param_env)
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
        // The body might be a constant, so it doesn't have codegen attributes.
        let self_features = &self.tcx.body_codegen_attrs(self.body_did.to_def_id()).target_features;

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

/// Context information for [`UnusedUnsafeVisitor`] traversal,
/// saves (innermost) relevant context
#[derive(Copy, Clone, Debug)]
enum Context {
    Safe,
    /// in an `unsafe fn`
    UnsafeFn(HirId),
    /// in a *used* `unsafe` block
    /// (i.e. a block without unused-unsafe warning)
    UnsafeBlock(HirId),
}

struct UnusedUnsafeVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    used_unsafe_blocks: &'a FxHashSet<HirId>,
    context: Context,
    unused_unsafes: &'a mut Vec<(HirId, UnusedUnsafe)>,
}

impl<'tcx> intravisit::Visitor<'tcx> for UnusedUnsafeVisitor<'_, 'tcx> {
    fn visit_block(&mut self, block: &'tcx hir::Block<'tcx>) {
        if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = block.rules {
            let used = match self.tcx.lint_level_at_node(UNUSED_UNSAFE, block.hir_id) {
                (Level::Allow, _) => true,
                _ => self.used_unsafe_blocks.contains(&block.hir_id),
            };
            let unused_unsafe = match (self.context, used) {
                (_, false) => UnusedUnsafe::Unused,
                (Context::Safe, true) | (Context::UnsafeFn(_), true) => {
                    let previous_context = self.context;
                    self.context = Context::UnsafeBlock(block.hir_id);
                    intravisit::walk_block(self, block);
                    self.context = previous_context;
                    return;
                }
                (Context::UnsafeBlock(hir_id), true) => UnusedUnsafe::InUnsafeBlock(hir_id),
            };
            self.unused_unsafes.push((block.hir_id, unused_unsafe));
        }
        intravisit::walk_block(self, block);
    }

    fn visit_anon_const(&mut self, c: &'tcx hir::AnonConst) {
        if matches!(self.tcx.def_kind(c.def_id), DefKind::InlineConst) {
            self.visit_body(self.tcx.hir().body(c.body))
        }
    }

    fn visit_fn(
        &mut self,
        fk: intravisit::FnKind<'tcx>,
        _fd: &'tcx hir::FnDecl<'tcx>,
        b: hir::BodyId,
        _s: rustc_span::Span,
        _id: LocalDefId,
    ) {
        if matches!(fk, intravisit::FnKind::Closure) {
            self.visit_body(self.tcx.hir().body(b))
        }
    }
}

fn check_unused_unsafe(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    used_unsafe_blocks: &FxHashSet<HirId>,
) -> Vec<(HirId, UnusedUnsafe)> {
    let body_id = tcx.hir().maybe_body_owned_by(def_id);

    let Some(body_id) = body_id else {
        debug!("check_unused_unsafe({:?}) - no body found", def_id);
        return vec![];
    };

    let body = tcx.hir().body(body_id);
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let context = match tcx.hir().fn_sig_by_hir_id(hir_id) {
        Some(sig) if sig.header.unsafety == hir::Unsafety::Unsafe => Context::UnsafeFn(hir_id),
        _ => Context::Safe,
    };

    debug!(
        "check_unused_unsafe({:?}, context={:?}, body={:?}, used_unsafe_blocks={:?})",
        def_id, body, context, used_unsafe_blocks
    );

    let mut unused_unsafes = vec![];

    let mut visitor = UnusedUnsafeVisitor {
        tcx,
        used_unsafe_blocks,
        context,
        unused_unsafes: &mut unused_unsafes,
    };
    intravisit::Visitor::visit_body(&mut visitor, body);

    unused_unsafes
}

fn unsafety_check_result(
    tcx: TyCtxt<'_>,
    def: ty::WithOptConstParam<LocalDefId>,
) -> &UnsafetyCheckResult {
    debug!("unsafety_violations({:?})", def);

    // N.B., this borrow is valid because all the consumers of
    // `mir_built` force this.
    let body = &tcx.mir_built(def).borrow();

    if body.is_custom_mir() {
        return tcx.arena.alloc(UnsafetyCheckResult {
            violations: Vec::new(),
            used_unsafe_blocks: FxHashSet::default(),
            unused_unsafes: Some(Vec::new()),
        });
    }

    let param_env = tcx.param_env(def.did);

    let mut checker = UnsafetyChecker::new(body, def.did, tcx, param_env);
    checker.visit_body(&body);

    let unused_unsafes = (!tcx.is_typeck_child(def.did.to_def_id()))
        .then(|| check_unused_unsafe(tcx, def.did, &checker.used_unsafe_blocks));

    tcx.arena.alloc(UnsafetyCheckResult {
        violations: checker.violations,
        used_unsafe_blocks: checker.used_unsafe_blocks,
        unused_unsafes,
    })
}

fn report_unused_unsafe(tcx: TyCtxt<'_>, kind: UnusedUnsafe, id: HirId) {
    let span = tcx.sess.source_map().guess_head_span(tcx.hir().span(id));
    let msg = "unnecessary `unsafe` block";
    tcx.struct_span_lint_hir(UNUSED_UNSAFE, id, span, msg, |lint| {
        lint.span_label(span, msg);
        match kind {
            UnusedUnsafe::Unused => {}
            UnusedUnsafe::InUnsafeBlock(id) => {
                lint.span_label(
                    tcx.sess.source_map().guess_head_span(tcx.hir().span(id)),
                    "because it's nested under this `unsafe` block",
                );
            }
        }

        lint
    });
}

pub fn check_unsafety(tcx: TyCtxt<'_>, def_id: LocalDefId) {
    debug!("check_unsafety({:?})", def_id);

    // closures and inline consts are handled by their parent fn.
    if tcx.is_typeck_child(def_id.to_def_id()) {
        return;
    }

    let UnsafetyCheckResult { violations, unused_unsafes, .. } = tcx.unsafety_check_result(def_id);

    for &UnsafetyViolation { source_info, lint_root, kind, details } in violations.iter() {
        let (description, note) = details.description_and_note();

        match kind {
            UnsafetyViolationKind::General => {
                // once
                let unsafe_fn_msg = if unsafe_op_in_unsafe_fn_allowed(tcx, lint_root) {
                    " function or"
                } else {
                    ""
                };

                let mut err = struct_span_err!(
                    tcx.sess,
                    source_info.span,
                    E0133,
                    "{} is unsafe and requires unsafe{} block",
                    description,
                    unsafe_fn_msg,
                );
                err.span_label(source_info.span, description).note(note);
                let note_non_inherited = tcx.hir().parent_iter(lint_root).find(|(id, node)| {
                    if let Node::Expr(block) = node
                        && let ExprKind::Block(block, _) = block.kind
                        && let BlockCheckMode::UnsafeBlock(_) = block.rules
                    {
                        true
                    }
                    else if let Some(sig) = tcx.hir().fn_sig_by_hir_id(*id)
                        && sig.header.is_unsafe()
                    {
                        true
                    } else {
                        false
                    }
                });
                if let Some((id, _)) = note_non_inherited {
                    let span = tcx.hir().span(id);
                    err.span_label(
                        tcx.sess.source_map().guess_head_span(span),
                        "items do not inherit unsafety from separate enclosing items",
                    );
                }

                err.emit();
            }
            UnsafetyViolationKind::UnsafeFn => tcx.struct_span_lint_hir(
                UNSAFE_OP_IN_UNSAFE_FN,
                lint_root,
                source_info.span,
                format!("{} is unsafe and requires unsafe block (error E0133)", description,),
                |lint| lint.span_label(source_info.span, description).note(note),
            ),
        }
    }

    for &(block_id, kind) in unused_unsafes.as_ref().unwrap() {
        report_unused_unsafe(tcx, kind, block_id);
    }
}

fn unsafe_op_in_unsafe_fn_allowed(tcx: TyCtxt<'_>, id: HirId) -> bool {
    tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, id).0 == Level::Allow
}
