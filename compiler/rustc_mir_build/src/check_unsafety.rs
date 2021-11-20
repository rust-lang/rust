use crate::build::ExprCategory;
use rustc_middle::thir::visit::{self, Visitor};

use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_middle::mir::BorrowKind;
use rustc_middle::thir::*;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_session::lint::builtin::{UNSAFE_OP_IN_UNSAFE_FN, UNUSED_UNSAFE};
use rustc_session::lint::Level;
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

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
    body_unsafety: BodyUnsafety,
    /// The `#[target_feature]` attributes of the body. Used for checking
    /// calls to functions with `#[target_feature]` (RFC 2396).
    body_target_features: &'tcx Vec<Symbol>,
    /// When inside the LHS of an assignment to a field, this is the type
    /// of the LHS and the span of the assignment expression.
    assignment_info: Option<(Ty<'tcx>, Span)>,
    in_union_destructure: bool,
    param_env: ParamEnv<'tcx>,
    inside_adt: bool,
}

impl<'tcx> UnsafetyVisitor<'_, 'tcx> {
    fn in_safety_context(&mut self, safety_context: SafetyContext, f: impl FnOnce(&mut Self)) {
        if let (
            SafetyContext::UnsafeBlock { span: enclosing_span, .. },
            SafetyContext::UnsafeBlock { span: block_span, hir_id, .. },
        ) = (self.safety_context, safety_context)
        {
            self.warn_unused_unsafe(
                hir_id,
                block_span,
                Some((self.tcx.sess.source_map().guess_head_span(enclosing_span), "block")),
            );
            f(self);
        } else {
            let prev_context = self.safety_context;
            self.safety_context = safety_context;

            f(self);

            if let SafetyContext::UnsafeBlock { used: false, span, hir_id } = self.safety_context {
                self.warn_unused_unsafe(
                    hir_id,
                    span,
                    if self.unsafe_op_in_unsafe_fn_allowed() {
                        self.body_unsafety.unsafe_fn_sig_span().map(|span| (span, "fn"))
                    } else {
                        None
                    },
                );
            }
            self.safety_context = prev_context;
        }
    }

    fn requires_unsafe(&mut self, span: Span, kind: UnsafeOpKind) {
        let (description, note) = kind.description_and_note();
        let unsafe_op_in_unsafe_fn_allowed = self.unsafe_op_in_unsafe_fn_allowed();
        match self.safety_context {
            SafetyContext::BuiltinUnsafeBlock => {}
            SafetyContext::UnsafeBlock { ref mut used, .. } => {
                if !self.body_unsafety.is_unsafe() || !unsafe_op_in_unsafe_fn_allowed {
                    // Mark this block as useful
                    *used = true;
                }
            }
            SafetyContext::UnsafeFn if unsafe_op_in_unsafe_fn_allowed => {}
            SafetyContext::UnsafeFn => {
                // unsafe_op_in_unsafe_fn is disallowed
                self.tcx.struct_span_lint_hir(
                    UNSAFE_OP_IN_UNSAFE_FN,
                    self.hir_context,
                    span,
                    |lint| {
                        lint.build(&format!(
                            "{} is unsafe and requires unsafe block (error E0133)",
                            description,
                        ))
                        .span_label(span, description)
                        .note(note)
                        .emit();
                    },
                )
            }
            SafetyContext::Safe => {
                let fn_sugg = if unsafe_op_in_unsafe_fn_allowed { " function or" } else { "" };
                struct_span_err!(
                    self.tcx.sess,
                    span,
                    E0133,
                    "{} is unsafe and requires unsafe{} block",
                    description,
                    fn_sugg,
                )
                .span_label(span, description)
                .note(note)
                .emit();
            }
        }
    }

    fn warn_unused_unsafe(
        &self,
        hir_id: hir::HirId,
        block_span: Span,
        enclosing_unsafe: Option<(Span, &'static str)>,
    ) {
        let block_span = self.tcx.sess.source_map().guess_head_span(block_span);
        self.tcx.struct_span_lint_hir(UNUSED_UNSAFE, hir_id, block_span, |lint| {
            let msg = "unnecessary `unsafe` block";
            let mut db = lint.build(msg);
            db.span_label(block_span, msg);
            if let Some((span, kind)) = enclosing_unsafe {
                db.span_label(span, format!("because it's nested under this `unsafe` {}", kind));
            }
            db.emit();
        });
    }

    /// Whether the `unsafe_op_in_unsafe_fn` lint is `allow`ed at the current HIR node.
    fn unsafe_op_in_unsafe_fn_allowed(&self) -> bool {
        self.tcx.lint_level_at_node(UNSAFE_OP_IN_UNSAFE_FN, self.hir_context).0 == Level::Allow
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

    fn visit_expr(&mut self, expr: &Expr<'tcx>) {
        match expr.kind {
            ExprKind::Field { lhs, .. } => {
                if let ty::Adt(adt_def, _) = self.thir[lhs].ty.kind() {
                    if (Bound::Unbounded, Bound::Unbounded)
                        != self.tcx.layout_scalar_valid_range(adt_def.did)
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
        &self.thir
    }

    fn visit_block(&mut self, block: &Block) {
        match block.safety_mode {
            // compiler-generated unsafe code should not count towards the usefulness of
            // an outer unsafe block
            BlockSafety::BuiltinUnsafe => {
                self.in_safety_context(SafetyContext::BuiltinUnsafeBlock, |this| {
                    visit::walk_block(this, block)
                });
            }
            BlockSafety::ExplicitUnsafe(hir_id) => {
                self.in_safety_context(
                    SafetyContext::UnsafeBlock { span: block.span, hir_id, used: false },
                    |this| visit::walk_block(this, block),
                );
            }
            BlockSafety::Safe => {
                visit::walk_block(self, block);
            }
        }
    }

    fn visit_pat(&mut self, pat: &Pat<'tcx>) {
        if self.in_union_destructure {
            match *pat.kind {
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
                // wildcard doesn't take anything
                PatKind::Wild |
                // these just wrap other patterns
                PatKind::Or { .. } |
                PatKind::AscribeUserType { .. } => {}
            }
        };

        match &*pat.kind {
            PatKind::Leaf { .. } => {
                if let ty::Adt(adt_def, ..) = pat.ty.kind() {
                    if adt_def.is_union() {
                        let old_in_union_destructure =
                            std::mem::replace(&mut self.in_union_destructure, true);
                        visit::walk_pat(self, pat);
                        self.in_union_destructure = old_in_union_destructure;
                    } else if (Bound::Unbounded, Bound::Unbounded)
                        != self.tcx.layout_scalar_valid_range(adt_def.did)
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
                        BorrowKind::Shallow | BorrowKind::Shared | BorrowKind::Unique => {
                            if !ty.is_freeze(self.tcx.at(pat.span), self.param_env) {
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
            _ => {
                visit::walk_pat(self, pat);
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr<'tcx>) {
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
            | ExprKind::ConstBlock { .. }
            | ExprKind::Deref { .. }
            | ExprKind::Index { .. }
            | ExprKind::NeverToAny { .. }
            | ExprKind::PlaceTypeAscription { .. }
            | ExprKind::ValueTypeAscription { .. }
            | ExprKind::Pointer { .. }
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
            | ExprKind::Yield { .. }
            | ExprKind::Loop { .. }
            | ExprKind::Let { .. }
            | ExprKind::Match { .. }
            | ExprKind::Box { .. }
            | ExprKind::If { .. }
            | ExprKind::InlineAsm { .. }
            | ExprKind::LlvmInlineAsm { .. }
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
                    self.requires_unsafe(expr.span, CallToUnsafeFunction);
                } else if let &ty::FnDef(func_did, _) = self.thir[fun].ty.kind() {
                    // If the called function has target features the calling function hasn't,
                    // the call requires `unsafe`. Don't check this on wasm
                    // targets, though. For more information on wasm see the
                    // is_like_wasm check in typeck/src/collect.rs
                    if !self.tcx.sess.target.options.is_like_wasm
                        && !self
                            .tcx
                            .codegen_fn_attrs(func_did)
                            .target_features
                            .iter()
                            .all(|feature| self.body_target_features.contains(feature))
                    {
                        self.requires_unsafe(expr.span, CallToFunctionWith);
                    }
                }
            }
            ExprKind::Deref { arg } => {
                if let ExprKind::StaticRef { def_id, .. } = self.thir[arg].kind {
                    if self.tcx.is_mutable_static(def_id) {
                        self.requires_unsafe(expr.span, UseOfMutableStatic);
                    } else if self.tcx.is_foreign_item(def_id) {
                        self.requires_unsafe(expr.span, UseOfExternStatic);
                    }
                } else if self.thir[arg].ty.is_unsafe_ptr() {
                    self.requires_unsafe(expr.span, DerefOfRawPointer);
                }
            }
            ExprKind::InlineAsm { .. } | ExprKind::LlvmInlineAsm { .. } => {
                self.requires_unsafe(expr.span, UseOfInlineAssembly);
            }
            ExprKind::Adt(box Adt {
                adt_def,
                variant_index: _,
                substs: _,
                user_ty: _,
                fields: _,
                base: _,
            }) => match self.tcx.layout_scalar_valid_range(adt_def.did) {
                (Bound::Unbounded, Bound::Unbounded) => {}
                _ => self.requires_unsafe(expr.span, InitializingTypeWith),
            },
            ExprKind::Closure {
                closure_id,
                substs: _,
                upvars: _,
                movability: _,
                fake_reads: _,
            } => {
                let closure_id = closure_id.expect_local();
                let closure_def = if let Some((did, const_param_id)) =
                    ty::WithOptConstParam::try_lookup(closure_id, self.tcx)
                {
                    ty::WithOptConstParam { did, const_param_did: Some(const_param_id) }
                } else {
                    ty::WithOptConstParam::unknown(closure_id)
                };
                let (closure_thir, expr) = self.tcx.thir_body(closure_def);
                let closure_thir = &closure_thir.borrow();
                let hir_context = self.tcx.hir().local_def_id_to_hir_id(closure_id);
                let mut closure_visitor =
                    UnsafetyVisitor { thir: closure_thir, hir_context, ..*self };
                closure_visitor.visit_expr(&closure_thir[expr]);
                // Unsafe blocks can be used in closures, make sure to take it into account
                self.safety_context = closure_visitor.safety_context;
            }
            ExprKind::Field { lhs, .. } => {
                let lhs = &self.thir[lhs];
                if let ty::Adt(adt_def, _) = lhs.ty.kind() {
                    if adt_def.is_union() {
                        if let Some((assigned_ty, assignment_span)) = self.assignment_info {
                            // To avoid semver hazard, we only consider `Copy` and `ManuallyDrop` non-dropping.
                            if !(assigned_ty
                                .ty_adt_def()
                                .map_or(false, |adt| adt.is_manually_drop())
                                || assigned_ty
                                    .is_copy_modulo_regions(self.tcx.at(expr.span), self.param_env))
                            {
                                self.requires_unsafe(assignment_span, AssignToDroppingUnionField);
                            } else {
                                // write to non-drop union field, safe
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

                // Second, check for accesses to union fields
                // don't have any special handling for AssignOp since it causes a read *and* write to lhs
                if matches!(expr.kind, ExprKind::Assign { .. }) {
                    self.assignment_info = Some((lhs.ty, expr.span));
                    visit::walk_expr(self, lhs);
                    self.assignment_info = None;
                    visit::walk_expr(self, &self.thir()[rhs]);
                    return; // we have already visited everything by now
                }
            }
            ExprKind::Borrow { borrow_kind, arg } => {
                let mut visitor = LayoutConstrainedPlaceVisitor::new(self.thir, self.tcx);
                visit::walk_expr(&mut visitor, expr);
                if visitor.found {
                    match borrow_kind {
                        BorrowKind::Shallow | BorrowKind::Shared | BorrowKind::Unique
                            if !self.thir[arg]
                                .ty
                                .is_freeze(self.tcx.at(self.thir[arg].span), self.param_env) =>
                        {
                            self.requires_unsafe(expr.span, BorrowOfLayoutConstrainedField)
                        }
                        BorrowKind::Mut { .. } => {
                            self.requires_unsafe(expr.span, MutationOfLayoutConstrainedField)
                        }
                        BorrowKind::Shallow | BorrowKind::Shared | BorrowKind::Unique => {}
                    }
                }
            }
            ExprKind::Let { expr: expr_id, .. } => {
                let let_expr = &self.thir[expr_id];
                if let ty::Adt(adt_def, _) = let_expr.ty.kind() {
                    if adt_def.is_union() {
                        self.requires_unsafe(expr.span, AccessToUnionField);
                    }
                }
            }
            _ => {}
        }
        visit::walk_expr(self, expr);
    }
}

#[derive(Clone, Copy)]
enum SafetyContext {
    Safe,
    BuiltinUnsafeBlock,
    UnsafeFn,
    UnsafeBlock { span: Span, hir_id: hir::HirId, used: bool },
}

#[derive(Clone, Copy)]
enum BodyUnsafety {
    /// The body is not unsafe.
    Safe,
    /// The body is an unsafe function. The span points to
    /// the signature of the function.
    Unsafe(Span),
}

impl BodyUnsafety {
    /// Returns whether the body is unsafe.
    fn is_unsafe(&self) -> bool {
        matches!(self, BodyUnsafety::Unsafe(_))
    }

    /// If the body is unsafe, returns the `Span` of its signature.
    fn unsafe_fn_sig_span(self) -> Option<Span> {
        match self {
            BodyUnsafety::Unsafe(span) => Some(span),
            BodyUnsafety::Safe => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum UnsafeOpKind {
    CallToUnsafeFunction,
    UseOfInlineAssembly,
    InitializingTypeWith,
    UseOfMutableStatic,
    UseOfExternStatic,
    DerefOfRawPointer,
    AssignToDroppingUnionField,
    AccessToUnionField,
    MutationOfLayoutConstrainedField,
    BorrowOfLayoutConstrainedField,
    CallToFunctionWith,
}

use UnsafeOpKind::*;

impl UnsafeOpKind {
    pub fn description_and_note(&self) -> (&'static str, &'static str) {
        match self {
            CallToUnsafeFunction => (
                "call to unsafe function",
                "consult the function's documentation for information on how to avoid undefined \
                 behavior",
            ),
            UseOfInlineAssembly => (
                "use of inline assembly",
                "inline assembly is entirely unchecked and can cause undefined behavior",
            ),
            InitializingTypeWith => (
                "initializing type with `rustc_layout_scalar_valid_range` attr",
                "initializing a layout restricted type's field with a value outside the valid \
                 range is undefined behavior",
            ),
            UseOfMutableStatic => (
                "use of mutable static",
                "mutable statics can be mutated by multiple threads: aliasing violations or data \
                 races will cause undefined behavior",
            ),
            UseOfExternStatic => (
                "use of extern static",
                "extern statics are not controlled by the Rust type system: invalid data, \
                 aliasing violations or data races will cause undefined behavior",
            ),
            DerefOfRawPointer => (
                "dereference of raw pointer",
                "raw pointers may be null, dangling or unaligned; they can violate aliasing rules \
                 and cause data races: all of these are undefined behavior",
            ),
            AssignToDroppingUnionField => (
                "assignment to union field that might need dropping",
                "the previous content of the field will be dropped, which causes undefined \
                 behavior if the field was not properly initialized",
            ),
            AccessToUnionField => (
                "access to union field",
                "the field may not be properly initialized: using uninitialized data will cause \
                 undefined behavior",
            ),
            MutationOfLayoutConstrainedField => (
                "mutation of layout constrained field",
                "mutating layout constrained fields cannot statically be checked for valid values",
            ),
            BorrowOfLayoutConstrainedField => (
                "borrow of layout constrained field with interior mutability",
                "references to fields of layout constrained fields lose the constraints. Coupled \
                 with interior mutability, the field can be changed to invalid values",
            ),
            CallToFunctionWith => (
                "call to function with `#[target_feature]`",
                "can only be called if the required target features are available",
            ),
        }
    }
}

pub fn check_unsafety<'tcx>(tcx: TyCtxt<'tcx>, def: ty::WithOptConstParam<LocalDefId>) {
    // THIR unsafeck is gated under `-Z thir-unsafeck`
    if !tcx.sess.opts.debugging_opts.thir_unsafeck {
        return;
    }

    // Closures are handled by their owner, if it has a body
    if tcx.is_closure(def.did.to_def_id()) {
        let hir = tcx.hir();
        let owner = hir.enclosing_body_owner(hir.local_def_id_to_hir_id(def.did));
        tcx.ensure().thir_check_unsafety(hir.local_def_id(owner));
        return;
    }

    let (thir, expr) = tcx.thir_body(def);
    let thir = &thir.borrow();
    // If `thir` is empty, a type error occured, skip this body.
    if thir.exprs.is_empty() {
        return;
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def.did);
    let body_unsafety = tcx.hir().fn_sig_by_hir_id(hir_id).map_or(BodyUnsafety::Safe, |fn_sig| {
        if fn_sig.header.unsafety == hir::Unsafety::Unsafe {
            BodyUnsafety::Unsafe(fn_sig.span)
        } else {
            BodyUnsafety::Safe
        }
    });
    let body_target_features = &tcx.codegen_fn_attrs(def.did).target_features;
    let safety_context =
        if body_unsafety.is_unsafe() { SafetyContext::UnsafeFn } else { SafetyContext::Safe };
    let mut visitor = UnsafetyVisitor {
        tcx,
        thir,
        safety_context,
        hir_context: hir_id,
        body_unsafety,
        body_target_features,
        assignment_info: None,
        in_union_destructure: false,
        param_env: tcx.param_env(def.did),
        inside_adt: false,
    };
    visitor.visit_expr(&thir[expr]);
}

crate fn thir_check_unsafety<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) {
    if let Some(def) = ty::WithOptConstParam::try_lookup(def_id, tcx) {
        tcx.thir_check_unsafety_for_const_arg(def)
    } else {
        check_unsafety(tcx, ty::WithOptConstParam::unknown(def_id))
    }
}

crate fn thir_check_unsafety_for_const_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    (did, param_did): (LocalDefId, DefId),
) {
    check_unsafety(tcx, ty::WithOptConstParam { did, const_param_did: Some(param_did) })
}
