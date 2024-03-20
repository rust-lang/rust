use std::borrow::Cow;

use crate::build::ExprCategory;
use crate::errors::*;
use rustc_middle::thir::visit::Visitor;

use rustc_errors::DiagArgValue;
use rustc_hir as hir;
use rustc_middle::mir::BorrowKind;
use rustc_middle::thir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_session::lint::builtin::{UNSAFE_OP_IN_UNSAFE_FN, UNUSED_UNSAFE};
use rustc_session::lint::Level;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::Symbol;
use rustc_span::{sym, Span};

use std::mem;
use std::ops::Bound;

struct UnsafetyVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'a Thir<'tcx>,
    /// The `HirId` of the current scope, which would be the `HirId`
    /// of the current HIR node, modulo adjustments. Used for lint levels.
    hir_context: hir::HirId,
    /// The current "safety context". This notably tracks whether we are in an
    /// `unsafe` block, and whether it has been used.
    safety_context: SafetyContext,
    /// The `#[target_feature]` attributes of the body. Used for checking
    /// calls to functions with `#[target_feature]` (RFC 2396).
    body_target_features: &'tcx [Symbol],
    /// When inside the LHS of an assignment to a field, this is the type
    /// of the LHS and the span of the assignment expression.
    assignment_info: Option<Ty<'tcx>>,
    in_union_destructure: bool,
    param_env: ParamEnv<'tcx>,
    inside_adt: bool,
    warnings: &'a mut Vec<UnusedUnsafeWarning>,

    /// Flag to ensure that we only suggest wrapping the entire function body in
    /// an unsafe block once.
    suggest_unsafe_block: bool,
}

impl<'tcx> UnsafetyVisitor<'_, 'tcx> {
    fn in_safety_context(&mut self, safety_context: SafetyContext, f: impl FnOnce(&mut Self)) {
        let prev_context = mem::replace(&mut self.safety_context, safety_context);

        f(self);

        let safety_context = mem::replace(&mut self.safety_context, prev_context);
        if let SafetyContext::UnsafeBlock { used, span, hir_id, nested_used_blocks } =
            safety_context
        {
            if !used {
                self.warn_unused_unsafe(hir_id, span, None);

                if let SafetyContext::UnsafeBlock {
                    nested_used_blocks: ref mut prev_nested_used_blocks,
                    ..
                } = self.safety_context
                {
                    prev_nested_used_blocks.extend(nested_used_blocks);
                }
            } else {
                for block in nested_used_blocks {
                    self.warn_unused_unsafe(
                        block.hir_id,
                        block.span,
                        Some(UnusedUnsafeEnclosing::Block {
                            span: self.tcx.sess.source_map().guess_head_span(span),
                        }),
                    );
                }

                match self.safety_context {
                    SafetyContext::UnsafeBlock {
                        nested_used_blocks: ref mut prev_nested_used_blocks,
                        ..
                    } => {
                        prev_nested_used_blocks.push(NestedUsedBlock { hir_id, span });
                    }
                    _ => (),
                }
            }
        }
    }

    fn requires_unsafe(&mut self, span: Span, kind: UnsafeOpKind) {
        let unsafe_op_in_unsafe_fn_allowed = self.unsafe_op_in_unsafe_fn_allowed();
        match self.safety_context {
            SafetyContext::BuiltinUnsafeBlock => {}
            SafetyContext::UnsafeBlock { ref mut used, .. } => {
                // Mark this block as useful (even inside `unsafe fn`, where it is technically
                // redundant -- but we want to eventually enable `unsafe_op_in_unsafe_fn` by
                // default which will require those blocks:
                // https://github.com/rust-lang/rust/issues/71668#issuecomment-1203075594).
                *used = true;
            }
            SafetyContext::UnsafeFn if unsafe_op_in_unsafe_fn_allowed => {}
            SafetyContext::UnsafeFn => {
                // unsafe_op_in_unsafe_fn is disallowed
                kind.emit_unsafe_op_in_unsafe_fn_lint(
                    self.tcx,
                    self.hir_context,
                    span,
                    self.suggest_unsafe_block,
                );
                self.suggest_unsafe_block = false;
            }
            SafetyContext::Safe => {
                kind.emit_requires_unsafe_err(
                    self.tcx,
                    span,
                    self.hir_context,
                    unsafe_op_in_unsafe_fn_allowed,
                );
            }
        }
    }

    fn warn_unused_unsafe(
        &mut self,
        hir_id: hir::HirId,
        block_span: Span,
        enclosing_unsafe: Option<UnusedUnsafeEnclosing>,
    ) {
        self.warnings.push(UnusedUnsafeWarning { hir_id, block_span, enclosing_unsafe });
    }

    /// Whether the `unsafe_op_in_unsafe_fn` lint is `allow`ed at the current HIR node.
    fn unsafe_op_in_unsafe_fn_allowed(&self) -> bool {
        self.tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, self.hir_context).0 == Level::Allow
    }

    /// Handle closures/coroutines/inline-consts, which is unsafecked with their parent body.
    fn visit_inner_body(&mut self, def: LocalDefId) {
        if let Ok((inner_thir, expr)) = self.tcx.thir_body(def) {
            // Runs all other queries that depend on THIR.
            self.tcx.ensure_with_value().mir_built(def);
            let inner_thir = &inner_thir.steal();
            let hir_context = self.tcx.local_def_id_to_hir_id(def);
            let safety_context = mem::replace(&mut self.safety_context, SafetyContext::Safe);
            let mut inner_visitor = UnsafetyVisitor {
                tcx: self.tcx,
                thir: inner_thir,
                hir_context,
                safety_context,
                body_target_features: self.body_target_features,
                assignment_info: self.assignment_info,
                in_union_destructure: false,
                param_env: self.param_env,
                inside_adt: false,
                warnings: self.warnings,
                suggest_unsafe_block: self.suggest_unsafe_block,
            };
            inner_visitor.visit_expr(&inner_thir[expr]);
            // Unsafe blocks can be used in the inner body, make sure to take it into account
            self.safety_context = inner_visitor.safety_context;
        }
    }
}

// Searches for accesses to layout constrained fields.
struct LayoutConstrainedPlaceVisitor<'a, 'tcx> {
    found: bool,
    thir: &'a Thir<'tcx>,
    tcx: TyCtxt<'tcx>,
}

impl<'a, 'tcx> LayoutConstrainedPlaceVisitor<'a, 'tcx> {
    fn new(thir: &'a Thir<'tcx>, tcx: TyCtxt<'tcx>) -> Self {
        Self { found: false, thir, tcx }
    }
}

impl<'a, 'tcx> Visitor<'a, 'tcx> for LayoutConstrainedPlaceVisitor<'a, 'tcx> {
    fn thir(&self) -> &'a Thir<'tcx> {
        self.thir
    }

    fn visit_expr(&mut self, expr: &'a Expr<'tcx>) {
        match expr.kind {
            ExprKind::Field { lhs, .. } => {
                if let ty::Adt(adt_def, _) = self.thir[lhs].ty.kind() {
                    if (Bound::Unbounded, Bound::Unbounded)
                        != self.tcx.layout_scalar_valid_range(adt_def.did())
                    {
                        self.found = true;
                    }
                }
                visit::walk_expr(self, expr);
            }

            // Keep walking through the expression as long as we stay in the same
            // place, i.e. the expression is a place expression and not a dereference
            // (since dereferencing something leads us to a different place).
            ExprKind::Deref { .. } => {}
            ref kind if ExprCategory::of(kind).map_or(true, |cat| cat == ExprCategory::Place) => {
                visit::walk_expr(self, expr);
            }

            _ => {}
        }
    }
}

impl<'a, 'tcx> Visitor<'a, 'tcx> for UnsafetyVisitor<'a, 'tcx> {
    fn thir(&self) -> &'a Thir<'tcx> {
        self.thir
    }

    fn visit_block(&mut self, block: &'a Block) {
        match block.safety_mode {
            // compiler-generated unsafe code should not count towards the usefulness of
            // an outer unsafe block
            BlockSafety::BuiltinUnsafe => {
                self.in_safety_context(SafetyContext::BuiltinUnsafeBlock, |this| {
                    visit::walk_block(this, block)
                });
            }
            BlockSafety::ExplicitUnsafe(hir_id) => {
                let used =
                    matches!(self.tcx.lint_level_at_node(UNUSED_UNSAFE, hir_id), (Level::Allow, _));
                self.in_safety_context(
                    SafetyContext::UnsafeBlock {
                        span: block.span,
                        hir_id,
                        used,
                        nested_used_blocks: Vec::new(),
                    },
                    |this| visit::walk_block(this, block),
                );
            }
            BlockSafety::Safe => {
                visit::walk_block(self, block);
            }
        }
    }

    fn visit_pat(&mut self, pat: &'a Pat<'tcx>) {
        if self.in_union_destructure {
            match pat.kind {
                // binding to a variable allows getting stuff out of variable
                PatKind::Binding { .. }
                // match is conditional on having this value
                | PatKind::Constant { .. }
                | PatKind::Variant { .. }
                | PatKind::Leaf { .. }
                | PatKind::Deref { .. }
                | PatKind::Range { .. }
                | PatKind::Slice { .. }
                | PatKind::Array { .. } => {
                    self.requires_unsafe(pat.span, AccessToUnionField);
                    return; // we can return here since this already requires unsafe
                }
                // wildcard/never don't take anything
                PatKind::Wild |
                PatKind::Never |
                // these just wrap other patterns
                PatKind::Or { .. } |
                PatKind::InlineConstant { .. } |
                PatKind::AscribeUserType { .. } |
                PatKind::Error(_) => {}
            }
        };

        match &pat.kind {
            PatKind::Leaf { .. } => {
                if let ty::Adt(adt_def, ..) = pat.ty.kind() {
                    if adt_def.is_union() {
                        let old_in_union_destructure =
                            std::mem::replace(&mut self.in_union_destructure, true);
                        visit::walk_pat(self, pat);
                        self.in_union_destructure = old_in_union_destructure;
                    } else if (Bound::Unbounded, Bound::Unbounded)
                        != self.tcx.layout_scalar_valid_range(adt_def.did())
                    {
                        let old_inside_adt = std::mem::replace(&mut self.inside_adt, true);
                        visit::walk_pat(self, pat);
                        self.inside_adt = old_inside_adt;
                    } else {
                        visit::walk_pat(self, pat);
                    }
                } else {
                    visit::walk_pat(self, pat);
                }
            }
            PatKind::Binding { mode: BindingMode::ByRef(borrow_kind), ty, .. } => {
                if self.inside_adt {
                    let ty::Ref(_, ty, _) = ty.kind() else {
                        span_bug!(
                            pat.span,
                            "BindingMode::ByRef in pattern, but found non-reference type {}",
                            ty
                        );
                    };
                    match borrow_kind {
                        BorrowKind::Fake | BorrowKind::Shared => {
                            if !ty.is_freeze(self.tcx, self.param_env) {
                                self.requires_unsafe(pat.span, BorrowOfLayoutConstrainedField);
                            }
                        }
                        BorrowKind::Mut { .. } => {
                            self.requires_unsafe(pat.span, MutationOfLayoutConstrainedField);
                        }
                    }
                }
                visit::walk_pat(self, pat);
            }
            PatKind::Deref { .. } => {
                let old_inside_adt = std::mem::replace(&mut self.inside_adt, false);
                visit::walk_pat(self, pat);
                self.inside_adt = old_inside_adt;
            }
            PatKind::InlineConstant { def, .. } => {
                self.visit_inner_body(*def);
                visit::walk_pat(self, pat);
            }
            _ => {
                visit::walk_pat(self, pat);
            }
        }
    }

    fn visit_expr(&mut self, expr: &'a Expr<'tcx>) {
        // could we be in the LHS of an assignment to a field?
        match expr.kind {
            ExprKind::Field { .. }
            | ExprKind::VarRef { .. }
            | ExprKind::UpvarRef { .. }
            | ExprKind::Scope { .. }
            | ExprKind::Cast { .. } => {}

            ExprKind::AddressOf { .. }
            | ExprKind::Adt { .. }
            | ExprKind::Array { .. }
            | ExprKind::Binary { .. }
            | ExprKind::Block { .. }
            | ExprKind::Borrow { .. }
            | ExprKind::Literal { .. }
            | ExprKind::NamedConst { .. }
            | ExprKind::NonHirLiteral { .. }
            | ExprKind::ZstLiteral { .. }
            | ExprKind::ConstParam { .. }
            | ExprKind::ConstBlock { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Index { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::PointerCoercion { .. }
            | ExprKind::Repeat { .. }
            | ExprKind::StaticRef { .. }
            | ExprKind::ThreadLocalRef { .. }
            | ExprKind::Tuple { .. }
            | ExprKind::Unary { .. }
            | ExprKind::Call { .. }
            | ExprKind::Assign { .. }
            | ExprKind::AssignOp { .. }
            | ExprKind::Break { .. }
            | ExprKind::Closure { .. }
            | ExprKind::Continue { .. }
            | ExprKind::Return { .. }
            | ExprKind::Become { .. }
            | ExprKind::Yield { .. }
            | ExprKind::Loop { .. }
            | ExprKind::Let { .. }
            | ExprKind::Match { .. }
            | ExprKind::Box { .. }
            | ExprKind::If { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::OffsetOf { .. }
            | ExprKind::LogicalOp { .. }
            | ExprKind::Use { .. } => {
                // We don't need to save the old value and restore it
                // because all the place expressions can't have more
                // than one child.
                self.assignment_info = None;
            }
        };
        match expr.kind {
            ExprKind::Scope { value, lint_level: LintLevel::Explicit(hir_id), region_scope: _ } => {
                let prev_id = self.hir_context;
                self.hir_context = hir_id;
                self.visit_expr(&self.thir[value]);
                self.hir_context = prev_id;
                return; // don't visit the whole expression
            }
            ExprKind::Call { fun, ty: _, args: _, from_hir_call: _, fn_span: _ } => {
                if self.thir[fun].ty.fn_sig(self.tcx).unsafety() == hir::Unsafety::Unsafe {
                    let func_id = if let ty::FnDef(func_id, _) = self.thir[fun].ty.kind() {
                        Some(*func_id)
                    } else {
                        None
                    };
                    self.requires_unsafe(expr.span, CallToUnsafeFunction(func_id));
                } else if let &ty::FnDef(func_did, _) = self.thir[fun].ty.kind() {
                    // If the called function has target features the calling function hasn't,
                    // the call requires `unsafe`. Don't check this on wasm
                    // targets, though. For more information on wasm see the
                    // is_like_wasm check in hir_analysis/src/collect.rs
                    let callee_features = &self.tcx.codegen_fn_attrs(func_did).target_features;
                    if !self.tcx.sess.target.options.is_like_wasm
                        && !callee_features
                            .iter()
                            .all(|feature| self.body_target_features.contains(feature))
                    {
                        let missing: Vec<_> = callee_features
                            .iter()
                            .copied()
                            .filter(|feature| !self.body_target_features.contains(feature))
                            .collect();
                        let build_enabled = self
                            .tcx
                            .sess
                            .target_features
                            .iter()
                            .copied()
                            .filter(|feature| missing.contains(feature))
                            .collect();
                        self.requires_unsafe(
                            expr.span,
                            CallToFunctionWith { function: func_did, missing, build_enabled },
                        );
                    }
                }
            }
            ExprKind::Deref { arg } => {
                if let ExprKind::StaticRef { def_id, .. } | ExprKind::ThreadLocalRef(def_id) =
                    self.thir[arg].kind
                {
                    if self.tcx.is_mutable_static(def_id) {
                        self.requires_unsafe(expr.span, UseOfMutableStatic);
                    } else if self.tcx.is_foreign_item(def_id) {
                        self.requires_unsafe(expr.span, UseOfExternStatic);
                    }
                } else if self.thir[arg].ty.is_unsafe_ptr() {
                    self.requires_unsafe(expr.span, DerefOfRawPointer);
                }
            }
            ExprKind::InlineAsm { .. } => {
                self.requires_unsafe(expr.span, UseOfInlineAssembly);
            }
            ExprKind::Adt(box AdtExpr {
                adt_def,
                variant_index: _,
                args: _,
                user_ty: _,
                fields: _,
                base: _,
            }) => match self.tcx.layout_scalar_valid_range(adt_def.did()) {
                (Bound::Unbounded, Bound::Unbounded) => {}
                _ => self.requires_unsafe(expr.span, InitializingTypeWith),
            },
            ExprKind::Closure(box ClosureExpr {
                closure_id,
                args: _,
                upvars: _,
                movability: _,
                fake_reads: _,
            }) => {
                self.visit_inner_body(closure_id);
            }
            ExprKind::ConstBlock { did, args: _ } => {
                let def_id = did.expect_local();
                self.visit_inner_body(def_id);
            }
            ExprKind::Field { lhs, .. } => {
                let lhs = &self.thir[lhs];
                if let ty::Adt(adt_def, _) = lhs.ty.kind()
                    && adt_def.is_union()
                {
                    if let Some(assigned_ty) = self.assignment_info {
                        if assigned_ty.needs_drop(self.tcx, self.param_env) {
                            // This would be unsafe, but should be outright impossible since we
                            // reject such unions.
                            assert!(
                                self.tcx.dcx().has_errors().is_some(),
                                "union fields that need dropping should be impossible: \
                                {assigned_ty}"
                            );
                        }
                    } else {
                        self.requires_unsafe(expr.span, AccessToUnionField);
                    }
                }
            }
            ExprKind::Assign { lhs, rhs } | ExprKind::AssignOp { lhs, rhs, .. } => {
                let lhs = &self.thir[lhs];
                // First, check whether we are mutating a layout constrained field
                let mut visitor = LayoutConstrainedPlaceVisitor::new(self.thir, self.tcx);
                visit::walk_expr(&mut visitor, lhs);
                if visitor.found {
                    self.requires_unsafe(expr.span, MutationOfLayoutConstrainedField);
                }

                // Second, check for accesses to union fields. Don't have any
                // special handling for AssignOp since it causes a read *and*
                // write to lhs.
                if matches!(expr.kind, ExprKind::Assign { .. }) {
                    self.assignment_info = Some(lhs.ty);
                    visit::walk_expr(self, lhs);
                    self.assignment_info = None;
                    visit::walk_expr(self, &self.thir()[rhs]);
                    return; // We have already visited everything by now.
                }
            }
            ExprKind::Borrow { borrow_kind, arg } => {
                let mut visitor = LayoutConstrainedPlaceVisitor::new(self.thir, self.tcx);
                visit::walk_expr(&mut visitor, expr);
                if visitor.found {
                    match borrow_kind {
                        BorrowKind::Fake | BorrowKind::Shared
                            if !self.thir[arg].ty.is_freeze(self.tcx, self.param_env) =>
                        {
                            self.requires_unsafe(expr.span, BorrowOfLayoutConstrainedField)
                        }
                        BorrowKind::Mut { .. } => {
                            self.requires_unsafe(expr.span, MutationOfLayoutConstrainedField)
                        }
                        BorrowKind::Fake | BorrowKind::Shared => {}
                    }
                }
            }
            _ => {}
        }
        visit::walk_expr(self, expr);
    }
}

#[derive(Clone)]
enum SafetyContext {
    Safe,
    BuiltinUnsafeBlock,
    UnsafeFn,
    UnsafeBlock {
        span: Span,
        hir_id: hir::HirId,
        used: bool,
        nested_used_blocks: Vec<NestedUsedBlock>,
    },
}

#[derive(Clone, Copy)]
struct NestedUsedBlock {
    hir_id: hir::HirId,
    span: Span,
}

struct UnusedUnsafeWarning {
    hir_id: hir::HirId,
    block_span: Span,
    enclosing_unsafe: Option<UnusedUnsafeEnclosing>,
}

#[derive(Clone, PartialEq)]
enum UnsafeOpKind {
    CallToUnsafeFunction(Option<DefId>),
    UseOfInlineAssembly,
    InitializingTypeWith,
    UseOfMutableStatic,
    UseOfExternStatic,
    DerefOfRawPointer,
    AccessToUnionField,
    MutationOfLayoutConstrainedField,
    BorrowOfLayoutConstrainedField,
    CallToFunctionWith {
        function: DefId,
        /// Target features enabled in callee's `#[target_feature]` but missing in
        /// caller's `#[target_feature]`.
        missing: Vec<Symbol>,
        /// Target features in `missing` that are enabled at compile time
        /// (e.g., with `-C target-feature`).
        build_enabled: Vec<Symbol>,
    },
}

use UnsafeOpKind::*;

impl UnsafeOpKind {
    pub fn emit_unsafe_op_in_unsafe_fn_lint(
        &self,
        tcx: TyCtxt<'_>,
        hir_id: hir::HirId,
        span: Span,
        suggest_unsafe_block: bool,
    ) {
        let parent_id = tcx.hir().get_parent_item(hir_id);
        let parent_owner = tcx.hir_owner_node(parent_id);
        let should_suggest = parent_owner.fn_sig().is_some_and(|sig| sig.header.is_unsafe());
        let unsafe_not_inherited_note = if should_suggest {
            suggest_unsafe_block.then(|| {
                let body_span = tcx.hir().body(parent_owner.body_id().unwrap()).value.span;
                UnsafeNotInheritedLintNote {
                    signature_span: tcx.def_span(parent_id.def_id),
                    body_span,
                }
            })
        } else {
            None
        };
        // FIXME: ideally we would want to trim the def paths, but this is not
        // feasible with the current lint emission API (see issue #106126).
        match self {
            CallToUnsafeFunction(Some(did)) => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe {
                    span,
                    function: with_no_trimmed_paths!(tcx.def_path_str(*did)),
                    unsafe_not_inherited_note,
                },
            ),
            CallToUnsafeFunction(None) => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            UseOfInlineAssembly => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            InitializingTypeWith => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            UseOfMutableStatic => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            UseOfExternStatic => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            DerefOfRawPointer => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            AccessToUnionField => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            MutationOfLayoutConstrainedField => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            BorrowOfLayoutConstrainedField => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                },
            ),
            CallToFunctionWith { function, missing, build_enabled } => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe {
                    span,
                    function: with_no_trimmed_paths!(tcx.def_path_str(*function)),
                    missing_target_features: DiagArgValue::StrListSepByAnd(
                        missing.iter().map(|feature| Cow::from(feature.to_string())).collect(),
                    ),
                    missing_target_features_count: missing.len(),
                    note: if build_enabled.is_empty() { None } else { Some(()) },
                    build_target_features: DiagArgValue::StrListSepByAnd(
                        build_enabled
                            .iter()
                            .map(|feature| Cow::from(feature.to_string()))
                            .collect(),
                    ),
                    build_target_features_count: build_enabled.len(),
                    unsafe_not_inherited_note,
                },
            ),
        }
    }

    pub fn emit_requires_unsafe_err(
        &self,
        tcx: TyCtxt<'_>,
        span: Span,
        hir_context: hir::HirId,
        unsafe_op_in_unsafe_fn_allowed: bool,
    ) {
        let note_non_inherited = tcx.hir().parent_iter(hir_context).find(|(id, node)| {
            if let hir::Node::Expr(block) = node
                && let hir::ExprKind::Block(block, _) = block.kind
                && let hir::BlockCheckMode::UnsafeBlock(_) = block.rules
            {
                true
            } else if let Some(sig) = tcx.hir().fn_sig_by_hir_id(*id)
                && sig.header.is_unsafe()
            {
                true
            } else {
                false
            }
        });
        let unsafe_not_inherited_note = if let Some((id, _)) = note_non_inherited {
            let span = tcx.hir().span(id);
            let span = tcx.sess.source_map().guess_head_span(span);
            Some(UnsafeNotInheritedNote { span })
        } else {
            None
        };

        let dcx = tcx.dcx();
        match self {
            CallToUnsafeFunction(Some(did)) if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                    function: tcx.def_path_str(*did),
                });
            }
            CallToUnsafeFunction(Some(did)) => {
                dcx.emit_err(CallToUnsafeFunctionRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                    function: tcx.def_path_str(*did),
                });
            }
            CallToUnsafeFunction(None) if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            CallToUnsafeFunction(None) => {
                dcx.emit_err(CallToUnsafeFunctionRequiresUnsafeNameless {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfInlineAssembly if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfInlineAssembly => {
                dcx.emit_err(UseOfInlineAssemblyRequiresUnsafe { span, unsafe_not_inherited_note });
            }
            InitializingTypeWith if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            InitializingTypeWith => {
                dcx.emit_err(InitializingTypeWithRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfMutableStatic if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfMutableStatic => {
                dcx.emit_err(UseOfMutableStaticRequiresUnsafe { span, unsafe_not_inherited_note });
            }
            UseOfExternStatic if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfExternStatic => {
                dcx.emit_err(UseOfExternStaticRequiresUnsafe { span, unsafe_not_inherited_note });
            }
            DerefOfRawPointer if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            DerefOfRawPointer => {
                dcx.emit_err(DerefOfRawPointerRequiresUnsafe { span, unsafe_not_inherited_note });
            }
            AccessToUnionField if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            AccessToUnionField => {
                dcx.emit_err(AccessToUnionFieldRequiresUnsafe { span, unsafe_not_inherited_note });
            }
            MutationOfLayoutConstrainedField if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(
                    MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                        span,
                        unsafe_not_inherited_note,
                    },
                );
            }
            MutationOfLayoutConstrainedField => {
                dcx.emit_err(MutationOfLayoutConstrainedFieldRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            BorrowOfLayoutConstrainedField if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(
                    BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                        span,
                        unsafe_not_inherited_note,
                    },
                );
            }
            BorrowOfLayoutConstrainedField => {
                dcx.emit_err(BorrowOfLayoutConstrainedFieldRequiresUnsafe {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            CallToFunctionWith { function, missing, build_enabled }
                if unsafe_op_in_unsafe_fn_allowed =>
            {
                dcx.emit_err(CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    missing_target_features: DiagArgValue::StrListSepByAnd(
                        missing.iter().map(|feature| Cow::from(feature.to_string())).collect(),
                    ),
                    missing_target_features_count: missing.len(),
                    note: if build_enabled.is_empty() { None } else { Some(()) },
                    build_target_features: DiagArgValue::StrListSepByAnd(
                        build_enabled
                            .iter()
                            .map(|feature| Cow::from(feature.to_string()))
                            .collect(),
                    ),
                    build_target_features_count: build_enabled.len(),
                    unsafe_not_inherited_note,
                    function: tcx.def_path_str(*function),
                });
            }
            CallToFunctionWith { function, missing, build_enabled } => {
                dcx.emit_err(CallToFunctionWithRequiresUnsafe {
                    span,
                    missing_target_features: DiagArgValue::StrListSepByAnd(
                        missing.iter().map(|feature| Cow::from(feature.to_string())).collect(),
                    ),
                    missing_target_features_count: missing.len(),
                    note: if build_enabled.is_empty() { None } else { Some(()) },
                    build_target_features: DiagArgValue::StrListSepByAnd(
                        build_enabled
                            .iter()
                            .map(|feature| Cow::from(feature.to_string()))
                            .collect(),
                    ),
                    build_target_features_count: build_enabled.len(),
                    unsafe_not_inherited_note,
                    function: tcx.def_path_str(*function),
                });
            }
        }
    }
}

pub fn check_unsafety(tcx: TyCtxt<'_>, def: LocalDefId) {
    // THIR unsafeck can be disabled with `-Z thir-unsafeck=off`
    if !tcx.sess.opts.unstable_opts.thir_unsafeck {
        return;
    }

    // Closures and inline consts are handled by their owner, if it has a body
    // Also, don't safety check custom MIR
    if tcx.is_typeck_child(def.to_def_id()) || tcx.has_attr(def, sym::custom_mir) {
        return;
    }

    let Ok((thir, expr)) = tcx.thir_body(def) else { return };
    // Runs all other queries that depend on THIR.
    tcx.ensure_with_value().mir_built(def);
    let thir = &thir.steal();
    // If `thir` is empty, a type error occurred, skip this body.
    if thir.exprs.is_empty() {
        return;
    }

    let hir_id = tcx.local_def_id_to_hir_id(def);
    let safety_context = tcx.hir().fn_sig_by_hir_id(hir_id).map_or(SafetyContext::Safe, |fn_sig| {
        if fn_sig.header.unsafety == hir::Unsafety::Unsafe {
            SafetyContext::UnsafeFn
        } else {
            SafetyContext::Safe
        }
    });
    let body_target_features = &tcx.body_codegen_attrs(def.to_def_id()).target_features;
    let mut warnings = Vec::new();
    let mut visitor = UnsafetyVisitor {
        tcx,
        thir,
        safety_context,
        hir_context: hir_id,
        body_target_features,
        assignment_info: None,
        in_union_destructure: false,
        param_env: tcx.param_env(def),
        inside_adt: false,
        warnings: &mut warnings,
        suggest_unsafe_block: true,
    };
    visitor.visit_expr(&thir[expr]);

    warnings.sort_by_key(|w| w.block_span);
    for UnusedUnsafeWarning { hir_id, block_span, enclosing_unsafe } in warnings {
        let block_span = tcx.sess.source_map().guess_head_span(block_span);
        tcx.emit_node_span_lint(
            UNUSED_UNSAFE,
            hir_id,
            block_span,
            UnusedUnsafe { span: block_span, enclosing: enclosing_unsafe },
        );
    }
}
