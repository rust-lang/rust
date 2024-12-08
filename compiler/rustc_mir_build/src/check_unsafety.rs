use std::borrow::Cow;
use std::mem;
use std::ops::Bound;

use rustc_errors::DiagArgValue;
use rustc_hir::def::DefKind;
use rustc_hir::{self as hir, BindingMode, ByRef, HirId, Mutability, Safety};
use rustc_middle::middle::codegen_fn_attrs::TargetFeature;
use rustc_middle::mir::BorrowKind;
use rustc_middle::span_bug;
use rustc_middle::thir::visit::Visitor;
use rustc_middle::thir::*;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_session::lint::Level;
use rustc_session::lint::builtin::{DEPRECATED_SAFE_2024, UNSAFE_OP_IN_UNSAFE_FN, UNUSED_UNSAFE};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};

use crate::build::ExprCategory;
use crate::errors::*;

struct UnsafetyVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'a Thir<'tcx>,
    /// The `HirId` of the current scope, which would be the `HirId`
    /// of the current HIR node, modulo adjustments. Used for lint levels.
    hir_context: HirId,
    /// The current "safety context". This notably tracks whether we are in an
    /// `unsafe` block, and whether it has been used.
    safety_context: SafetyContext,
    /// The `#[target_feature]` attributes of the body. Used for checking
    /// calls to functions with `#[target_feature]` (RFC 2396).
    body_target_features: &'tcx [TargetFeature],
    /// When inside the LHS of an assignment to a field, this is the type
    /// of the LHS and the span of the assignment expression.
    assignment_info: Option<Ty<'tcx>>,
    in_union_destructure: bool,
    typing_env: ty::TypingEnv<'tcx>,
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

    fn emit_deprecated_safe_fn_call(&self, span: Span, kind: &UnsafeOpKind) -> bool {
        match kind {
            // Allow calls to deprecated-safe unsafe functions if the caller is
            // from an edition before 2024.
            &UnsafeOpKind::CallToUnsafeFunction(Some(id))
                if !span.at_least_rust_2024()
                    && let Some(attr) = self.tcx.get_attr(id, sym::rustc_deprecated_safe_2024) =>
            {
                let suggestion = attr
                    .meta_item_list()
                    .unwrap_or_default()
                    .into_iter()
                    .find(|item| item.has_name(sym::audit_that))
                    .map(|item| {
                        item.value_str().expect(
                            "`#[rustc_deprecated_safe_2024(audit_that)]` must have a string value",
                        )
                    });

                let sm = self.tcx.sess.source_map();
                let guarantee = suggestion
                    .as_ref()
                    .map(|suggestion| format!("that {}", suggestion))
                    .unwrap_or_else(|| String::from("its unsafe preconditions"));
                let suggestion = suggestion
                    .and_then(|suggestion| {
                        sm.indentation_before(span).map(|indent| {
                            format!("{}// TODO: Audit that {}.\n", indent, suggestion) // ignore-tidy-todo
                        })
                    })
                    .unwrap_or_default();

                self.tcx.emit_node_span_lint(
                    DEPRECATED_SAFE_2024,
                    self.hir_context,
                    span,
                    CallToDeprecatedSafeFnRequiresUnsafe {
                        span,
                        function: with_no_trimmed_paths!(self.tcx.def_path_str(id)),
                        guarantee,
                        sub: CallToDeprecatedSafeFnRequiresUnsafeSub {
                            start_of_line_suggestion: suggestion,
                            start_of_line: sm.span_extend_to_line(span).shrink_to_lo(),
                            left: span.shrink_to_lo(),
                            right: span.shrink_to_hi(),
                        },
                    },
                );
                true
            }
            _ => false,
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
                let deprecated_safe_fn = self.emit_deprecated_safe_fn_call(span, &kind);
                if !deprecated_safe_fn {
                    // unsafe_op_in_unsafe_fn is disallowed
                    kind.emit_unsafe_op_in_unsafe_fn_lint(
                        self.tcx,
                        self.hir_context,
                        span,
                        self.suggest_unsafe_block,
                    );
                    self.suggest_unsafe_block = false;
                }
            }
            SafetyContext::Safe => {
                let deprecated_safe_fn = self.emit_deprecated_safe_fn_call(span, &kind);
                if !deprecated_safe_fn {
                    kind.emit_requires_unsafe_err(
                        self.tcx,
                        span,
                        self.hir_context,
                        unsafe_op_in_unsafe_fn_allowed,
                    );
                }
            }
        }
    }

    fn warn_unused_unsafe(
        &mut self,
        hir_id: HirId,
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
                typing_env: self.typing_env,
                inside_adt: false,
                warnings: self.warnings,
                suggest_unsafe_block: self.suggest_unsafe_block,
            };
            // params in THIR may be unsafe, e.g. a union pattern.
            for param in &inner_thir.params {
                if let Some(param_pat) = param.pat.as_deref() {
                    inner_visitor.visit_pat(param_pat);
                }
            }
            // Visit the body.
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
                | PatKind::DerefPattern { .. }
                | PatKind::Range { .. }
                | PatKind::Slice { .. }
                | PatKind::Array { .. }
                // Never constitutes a witness of uninhabitedness.
                | PatKind::Never => {
                    self.requires_unsafe(pat.span, AccessToUnionField);
                    return; // we can return here since this already requires unsafe
                }
                // wildcard doesn't read anything.
                PatKind::Wild |
                // these just wrap other patterns, which we recurse on below.
                PatKind::Or { .. } |
                PatKind::ExpandedConstant { .. } |
                PatKind::AscribeUserType { .. } |
                PatKind::Error(_) => {}
            }
        };

        match &pat.kind {
            PatKind::Leaf { subpatterns, .. } => {
                if let ty::Adt(adt_def, ..) = pat.ty.kind() {
                    for pat in subpatterns {
                        if adt_def.non_enum_variant().fields[pat.field].safety == Safety::Unsafe {
                            self.requires_unsafe(pat.pattern.span, UseOfUnsafeField);
                        }
                    }
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
            PatKind::Variant { adt_def, args: _, variant_index, subpatterns } => {
                for pat in subpatterns {
                    let field = &pat.field;
                    if adt_def.variant(*variant_index).fields[*field].safety == Safety::Unsafe {
                        self.requires_unsafe(pat.pattern.span, UseOfUnsafeField);
                    }
                }
                visit::walk_pat(self, pat);
            }
            PatKind::Binding { mode: BindingMode(ByRef::Yes(rm), _), ty, .. } => {
                if self.inside_adt {
                    let ty::Ref(_, ty, _) = ty.kind() else {
                        span_bug!(
                            pat.span,
                            "ByRef::Yes in pattern, but found non-reference type {}",
                            ty
                        );
                    };
                    match rm {
                        Mutability::Not => {
                            if !ty.is_freeze(self.tcx, self.typing_env) {
                                self.requires_unsafe(pat.span, BorrowOfLayoutConstrainedField);
                            }
                        }
                        Mutability::Mut { .. } => {
                            self.requires_unsafe(pat.span, MutationOfLayoutConstrainedField);
                        }
                    }
                }
                visit::walk_pat(self, pat);
            }
            PatKind::Deref { .. } | PatKind::DerefPattern { .. } => {
                let old_inside_adt = std::mem::replace(&mut self.inside_adt, false);
                visit::walk_pat(self, pat);
                self.inside_adt = old_inside_adt;
            }
            PatKind::ExpandedConstant { def_id, is_inline, .. } => {
                if let Some(def) = def_id.as_local()
                    && *is_inline
                {
                    self.visit_inner_body(def);
                }
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

            ExprKind::RawBorrow { .. }
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
                if self.thir[fun].ty.fn_sig(self.tcx).safety() == hir::Safety::Unsafe {
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
                        && !callee_features.iter().all(|feature| {
                            self.body_target_features.iter().any(|f| f.name == feature.name)
                        })
                    {
                        let missing: Vec<_> = callee_features
                            .iter()
                            .copied()
                            .filter(|feature| {
                                !feature.implied
                                    && !self
                                        .body_target_features
                                        .iter()
                                        .any(|body_feature| body_feature.name == feature.name)
                            })
                            .map(|feature| feature.name)
                            .collect();
                        let build_enabled = self
                            .tcx
                            .sess
                            .target_features
                            .iter()
                            .copied()
                            .filter(|feature| missing.contains(feature))
                            .collect();
                        self.requires_unsafe(expr.span, CallToFunctionWith {
                            function: func_did,
                            missing,
                            build_enabled,
                        });
                    }
                }
            }
            ExprKind::RawBorrow { arg, .. } => {
                if let ExprKind::Scope { value: arg, .. } = self.thir[arg].kind
                    && let ExprKind::Deref { arg } = self.thir[arg].kind
                {
                    // Taking a raw ref to a deref place expr is always safe.
                    // Make sure the expression we're deref'ing is safe, though.
                    visit::walk_expr(self, &self.thir[arg]);
                    return;
                }
            }
            ExprKind::Deref { arg } => {
                if let ExprKind::StaticRef { def_id, .. } | ExprKind::ThreadLocalRef(def_id) =
                    self.thir[arg].kind
                {
                    if self.tcx.is_mutable_static(def_id) {
                        self.requires_unsafe(expr.span, UseOfMutableStatic);
                    } else if self.tcx.is_foreign_item(def_id) {
                        match self.tcx.def_kind(def_id) {
                            DefKind::Static { safety: hir::Safety::Safe, .. } => {}
                            _ => self.requires_unsafe(expr.span, UseOfExternStatic),
                        }
                    }
                } else if self.thir[arg].ty.is_unsafe_ptr() {
                    self.requires_unsafe(expr.span, DerefOfRawPointer);
                }
            }
            ExprKind::InlineAsm(box InlineAsmExpr {
                asm_macro: _,
                ref operands,
                template: _,
                options: _,
                line_spans: _,
            }) => {
                self.requires_unsafe(expr.span, UseOfInlineAssembly);

                // For inline asm, do not use `walk_expr`, since we want to handle the label block
                // specially.
                for op in &**operands {
                    use rustc_middle::thir::InlineAsmOperand::*;
                    match op {
                        In { expr, reg: _ }
                        | Out { expr: Some(expr), reg: _, late: _ }
                        | InOut { expr, reg: _, late: _ } => self.visit_expr(&self.thir()[*expr]),
                        SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                            self.visit_expr(&self.thir()[*in_expr]);
                            if let Some(out_expr) = out_expr {
                                self.visit_expr(&self.thir()[*out_expr]);
                            }
                        }
                        Out { expr: None, reg: _, late: _ }
                        | Const { value: _, span: _ }
                        | SymFn { value: _, span: _ }
                        | SymStatic { def_id: _ } => {}
                        Label { block } => {
                            // Label blocks are safe context.
                            // `asm!()` is forced to be wrapped inside unsafe. If there's no special
                            // treatment, the label blocks would also always be unsafe with no way
                            // of opting out.
                            self.in_safety_context(SafetyContext::Safe, |this| {
                                visit::walk_block(this, &this.thir()[*block])
                            });
                        }
                    }
                }
                return;
            }
            ExprKind::Adt(box AdtExpr {
                adt_def,
                variant_index,
                args: _,
                user_ty: _,
                fields: _,
                base: _,
            }) => {
                if adt_def.variant(variant_index).has_unsafe_fields() {
                    self.requires_unsafe(expr.span, InitializingTypeWithUnsafeField)
                }
                match self.tcx.layout_scalar_valid_range(adt_def.did()) {
                    (Bound::Unbounded, Bound::Unbounded) => {}
                    _ => self.requires_unsafe(expr.span, InitializingTypeWith),
                }
            }
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
            ExprKind::Field { lhs, variant_index, name } => {
                let lhs = &self.thir[lhs];
                if let ty::Adt(adt_def, _) = lhs.ty.kind() {
                    if adt_def.variant(variant_index).fields[name].safety == Safety::Unsafe {
                        self.requires_unsafe(expr.span, UseOfUnsafeField);
                    } else if adt_def.is_union() {
                        if let Some(assigned_ty) = self.assignment_info {
                            if assigned_ty.needs_drop(self.tcx, self.typing_env) {
                                // This would be unsafe, but should be outright impossible since we
                                // reject such unions.
                                assert!(
                                    self.tcx.dcx().has_errors().is_some(),
                                    "union fields that need dropping should be impossible: {assigned_ty}"
                                );
                            }
                        } else {
                            self.requires_unsafe(expr.span, AccessToUnionField);
                        }
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
                        BorrowKind::Fake(_) | BorrowKind::Shared
                            if !self.thir[arg].ty.is_freeze(self.tcx, self.typing_env) =>
                        {
                            self.requires_unsafe(expr.span, BorrowOfLayoutConstrainedField)
                        }
                        BorrowKind::Mut { .. } => {
                            self.requires_unsafe(expr.span, MutationOfLayoutConstrainedField)
                        }
                        BorrowKind::Fake(_) | BorrowKind::Shared => {}
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
    UnsafeBlock { span: Span, hir_id: HirId, used: bool, nested_used_blocks: Vec<NestedUsedBlock> },
}

#[derive(Clone, Copy)]
struct NestedUsedBlock {
    hir_id: HirId,
    span: Span,
}

struct UnusedUnsafeWarning {
    hir_id: HirId,
    block_span: Span,
    enclosing_unsafe: Option<UnusedUnsafeEnclosing>,
}

#[derive(Clone, PartialEq)]
enum UnsafeOpKind {
    CallToUnsafeFunction(Option<DefId>),
    UseOfInlineAssembly,
    InitializingTypeWith,
    InitializingTypeWithUnsafeField,
    UseOfMutableStatic,
    UseOfExternStatic,
    UseOfUnsafeField,
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
    fn emit_unsafe_op_in_unsafe_fn_lint(
        &self,
        tcx: TyCtxt<'_>,
        hir_id: HirId,
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
            InitializingTypeWithUnsafeField => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnInitializingTypeWithUnsafeFieldRequiresUnsafe {
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
            UseOfUnsafeField => tcx.emit_node_span_lint(
                UNSAFE_OP_IN_UNSAFE_FN,
                hir_id,
                span,
                UnsafeOpInUnsafeFnUseOfUnsafeFieldRequiresUnsafe {
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
                    note: !build_enabled.is_empty(),
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

    fn emit_requires_unsafe_err(
        &self,
        tcx: TyCtxt<'_>,
        span: Span,
        hir_context: HirId,
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
            InitializingTypeWithUnsafeField if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(
                    InitializingTypeWithUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                        span,
                        unsafe_not_inherited_note,
                    },
                );
            }
            InitializingTypeWithUnsafeField => {
                dcx.emit_err(InitializingTypeWithUnsafeFieldRequiresUnsafe {
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
            UseOfUnsafeField if unsafe_op_in_unsafe_fn_allowed => {
                dcx.emit_err(UseOfUnsafeFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed {
                    span,
                    unsafe_not_inherited_note,
                });
            }
            UseOfUnsafeField => {
                dcx.emit_err(UseOfUnsafeFieldRequiresUnsafe { span, unsafe_not_inherited_note });
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
                    note: !build_enabled.is_empty(),
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
                    note: !build_enabled.is_empty(),
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

pub(crate) fn check_unsafety(tcx: TyCtxt<'_>, def: LocalDefId) {
    // Closures and inline consts are handled by their owner, if it has a body
    // Also, don't safety check custom MIR
    if tcx.is_typeck_child(def.to_def_id()) || tcx.has_attr(def, sym::custom_mir) {
        return;
    }

    let Ok((thir, expr)) = tcx.thir_body(def) else { return };
    // Runs all other queries that depend on THIR.
    tcx.ensure_with_value().mir_built(def);
    let thir = &thir.steal();

    let hir_id = tcx.local_def_id_to_hir_id(def);
    let safety_context = tcx.hir().fn_sig_by_hir_id(hir_id).map_or(SafetyContext::Safe, |fn_sig| {
        if fn_sig.header.safety == hir::Safety::Unsafe {
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
        // FIXME(#132279): we're clearly in a body here.
        typing_env: ty::TypingEnv::non_body_analysis(tcx, def),
        inside_adt: false,
        warnings: &mut warnings,
        suggest_unsafe_block: true,
    };
    // params in THIR may be unsafe, e.g. a union pattern.
    for param in &thir.params {
        if let Some(param_pat) = param.pat.as_deref() {
            visitor.visit_pat(param_pat);
        }
    }
    // Visit the body.
    visitor.visit_expr(&thir[expr]);

    warnings.sort_by_key(|w| w.block_span);
    for UnusedUnsafeWarning { hir_id, block_span, enclosing_unsafe } in warnings {
        let block_span = tcx.sess.source_map().guess_head_span(block_span);
        tcx.emit_node_span_lint(UNUSED_UNSAFE, hir_id, block_span, UnusedUnsafe {
            span: block_span,
            enclosing: enclosing_unsafe,
        });
    }
}
