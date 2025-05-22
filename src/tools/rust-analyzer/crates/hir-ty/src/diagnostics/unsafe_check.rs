//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::mem;

use either::Either;
use hir_def::{
    AdtId, DefWithBodyId, FieldId, FunctionId, VariantId,
    expr_store::{Body, path::Path},
    hir::{AsmOperand, Expr, ExprId, ExprOrPatId, Pat, PatId, Statement, UnaryOp},
    resolver::{HasResolver, ResolveValueResult, Resolver, ValueNs},
    signatures::StaticFlags,
    type_ref::Rawness,
};
use span::Edition;

use crate::{
    InferenceResult, Interner, TargetFeatures, TyExt, TyKind, db::HirDatabase,
    utils::is_fn_unsafe_to_call,
};

#[derive(Debug, Default)]
pub struct MissingUnsafeResult {
    pub unsafe_exprs: Vec<(ExprOrPatId, UnsafetyReason)>,
    /// If `fn_is_unsafe` is false, `unsafe_exprs` are hard errors. If true, they're `unsafe_op_in_unsafe_fn`.
    pub fn_is_unsafe: bool,
    pub deprecated_safe_calls: Vec<ExprId>,
}

pub fn missing_unsafe(db: &dyn HirDatabase, def: DefWithBodyId) -> MissingUnsafeResult {
    let _p = tracing::info_span!("missing_unsafe").entered();

    let is_unsafe = match def {
        DefWithBodyId::FunctionId(it) => db.function_signature(it).is_unsafe(),
        DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_) | DefWithBodyId::VariantId(_) => {
            false
        }
    };

    let mut res = MissingUnsafeResult { fn_is_unsafe: is_unsafe, ..MissingUnsafeResult::default() };
    let body = db.body(def);
    let infer = db.infer(def);
    let mut callback = |diag| match diag {
        UnsafeDiagnostic::UnsafeOperation { node, inside_unsafe_block, reason } => {
            if inside_unsafe_block == InsideUnsafeBlock::No {
                res.unsafe_exprs.push((node, reason));
            }
        }
        UnsafeDiagnostic::DeprecatedSafe2024 { node, inside_unsafe_block } => {
            if inside_unsafe_block == InsideUnsafeBlock::No {
                res.deprecated_safe_calls.push(node)
            }
        }
    };
    let mut visitor = UnsafeVisitor::new(db, &infer, &body, def, &mut callback);
    visitor.walk_expr(body.body_expr);

    if !is_unsafe {
        // Unsafety in function parameter patterns (that can only be union destructuring)
        // cannot be inserted into an unsafe block, so even with `unsafe_op_in_unsafe_fn`
        // it is turned off for unsafe functions.
        for &param in &body.params {
            visitor.walk_pat(param);
        }
    }

    res
}

#[derive(Debug, Clone, Copy)]
pub enum UnsafetyReason {
    UnionField,
    UnsafeFnCall,
    InlineAsm,
    RawPtrDeref,
    MutableStatic,
    ExternStatic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsideUnsafeBlock {
    No,
    Yes,
}

#[derive(Debug)]
enum UnsafeDiagnostic {
    UnsafeOperation {
        node: ExprOrPatId,
        inside_unsafe_block: InsideUnsafeBlock,
        reason: UnsafetyReason,
    },
    /// A lint.
    DeprecatedSafe2024 { node: ExprId, inside_unsafe_block: InsideUnsafeBlock },
}

pub fn unsafe_operations_for_body(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    callback: &mut dyn FnMut(ExprOrPatId),
) {
    let mut visitor_callback = |diag| {
        if let UnsafeDiagnostic::UnsafeOperation { node, .. } = diag {
            callback(node);
        }
    };
    let mut visitor = UnsafeVisitor::new(db, infer, body, def, &mut visitor_callback);
    visitor.walk_expr(body.body_expr);
    for &param in &body.params {
        visitor.walk_pat(param);
    }
}

pub fn unsafe_operations(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    current: ExprId,
    callback: &mut dyn FnMut(InsideUnsafeBlock),
) {
    let mut visitor_callback = |diag| {
        if let UnsafeDiagnostic::UnsafeOperation { inside_unsafe_block, .. } = diag {
            callback(inside_unsafe_block);
        }
    };
    let mut visitor = UnsafeVisitor::new(db, infer, body, def, &mut visitor_callback);
    _ = visitor.resolver.update_to_inner_scope(db, def, current);
    visitor.walk_expr(current);
}

struct UnsafeVisitor<'db> {
    db: &'db dyn HirDatabase,
    infer: &'db InferenceResult,
    body: &'db Body,
    resolver: Resolver<'db>,
    def: DefWithBodyId,
    inside_unsafe_block: InsideUnsafeBlock,
    inside_assignment: bool,
    inside_union_destructure: bool,
    callback: &'db mut dyn FnMut(UnsafeDiagnostic),
    def_target_features: TargetFeatures,
    // FIXME: This needs to be the edition of the span of each call.
    edition: Edition,
}

impl<'db> UnsafeVisitor<'db> {
    fn new(
        db: &'db dyn HirDatabase,
        infer: &'db InferenceResult,
        body: &'db Body,
        def: DefWithBodyId,
        unsafe_expr_cb: &'db mut dyn FnMut(UnsafeDiagnostic),
    ) -> Self {
        let resolver = def.resolver(db);
        let def_target_features = match def {
            DefWithBodyId::FunctionId(func) => TargetFeatures::from_attrs(&db.attrs(func.into())),
            _ => TargetFeatures::default(),
        };
        let edition = resolver.module().krate().data(db).edition;
        Self {
            db,
            infer,
            body,
            resolver,
            def,
            inside_unsafe_block: InsideUnsafeBlock::No,
            inside_assignment: false,
            inside_union_destructure: false,
            callback: unsafe_expr_cb,
            def_target_features,
            edition,
        }
    }

    fn on_unsafe_op(&mut self, node: ExprOrPatId, reason: UnsafetyReason) {
        (self.callback)(UnsafeDiagnostic::UnsafeOperation {
            node,
            inside_unsafe_block: self.inside_unsafe_block,
            reason,
        });
    }

    fn check_call(&mut self, node: ExprId, func: FunctionId) {
        let unsafety = is_fn_unsafe_to_call(self.db, func, &self.def_target_features, self.edition);
        match unsafety {
            crate::utils::Unsafety::Safe => {}
            crate::utils::Unsafety::Unsafe => {
                self.on_unsafe_op(node.into(), UnsafetyReason::UnsafeFnCall)
            }
            crate::utils::Unsafety::DeprecatedSafe2024 => {
                (self.callback)(UnsafeDiagnostic::DeprecatedSafe2024 {
                    node,
                    inside_unsafe_block: self.inside_unsafe_block,
                })
            }
        }
    }

    fn with_inside_unsafe_block<R>(
        &mut self,
        inside_unsafe_block: InsideUnsafeBlock,
        f: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let old = mem::replace(&mut self.inside_unsafe_block, inside_unsafe_block);
        let result = f(self);
        self.inside_unsafe_block = old;
        result
    }

    fn walk_pats_top(&mut self, pats: impl Iterator<Item = PatId>, parent_expr: ExprId) {
        let guard = self.resolver.update_to_inner_scope(self.db, self.def, parent_expr);
        pats.for_each(|pat| self.walk_pat(pat));
        self.resolver.reset_to_guard(guard);
    }

    fn walk_pat(&mut self, current: PatId) {
        let pat = &self.body.pats[current];

        if self.inside_union_destructure {
            match pat {
                Pat::Tuple { .. }
                | Pat::Record { .. }
                | Pat::Range { .. }
                | Pat::Slice { .. }
                | Pat::Path(..)
                | Pat::Lit(..)
                | Pat::Bind { .. }
                | Pat::TupleStruct { .. }
                | Pat::Ref { .. }
                | Pat::Box { .. }
                | Pat::Expr(..)
                | Pat::ConstBlock(..) => {
                    self.on_unsafe_op(current.into(), UnsafetyReason::UnionField)
                }
                // `Or` only wraps other patterns, and `Missing`/`Wild` do not constitute a read.
                Pat::Missing | Pat::Wild | Pat::Or(_) => {}
            }
        }

        match pat {
            Pat::Record { .. } => {
                if let Some((AdtId::UnionId(_), _)) = self.infer[current].as_adt() {
                    let old_inside_union_destructure =
                        mem::replace(&mut self.inside_union_destructure, true);
                    self.body.walk_pats_shallow(current, |pat| self.walk_pat(pat));
                    self.inside_union_destructure = old_inside_union_destructure;
                    return;
                }
            }
            Pat::Path(path) => self.mark_unsafe_path(current.into(), path),
            &Pat::ConstBlock(expr) => {
                let old_inside_assignment = mem::replace(&mut self.inside_assignment, false);
                self.walk_expr(expr);
                self.inside_assignment = old_inside_assignment;
            }
            &Pat::Expr(expr) => self.walk_expr(expr),
            _ => {}
        }

        self.body.walk_pats_shallow(current, |pat| self.walk_pat(pat));
    }

    fn walk_expr(&mut self, current: ExprId) {
        let expr = &self.body.exprs[current];
        let inside_assignment = mem::replace(&mut self.inside_assignment, false);
        match expr {
            &Expr::Call { callee, .. } => {
                let callee = &self.infer[callee];
                if let Some(func) = callee.as_fn_def(self.db) {
                    self.check_call(current, func);
                }
                if let TyKind::Function(fn_ptr) = callee.kind(Interner) {
                    if fn_ptr.sig.safety == chalk_ir::Safety::Unsafe {
                        self.on_unsafe_op(current.into(), UnsafetyReason::UnsafeFnCall);
                    }
                }
            }
            Expr::Path(path) => {
                let guard = self.resolver.update_to_inner_scope(self.db, self.def, current);
                self.mark_unsafe_path(current.into(), path);
                self.resolver.reset_to_guard(guard);
            }
            Expr::Ref { expr, rawness: Rawness::RawPtr, mutability: _ } => {
                match self.body.exprs[*expr] {
                    // Do not report unsafe for `addr_of[_mut]!(EXTERN_OR_MUT_STATIC)`,
                    // see https://github.com/rust-lang/rust/pull/125834.
                    Expr::Path(_) => return,
                    // https://github.com/rust-lang/rust/pull/129248
                    // Taking a raw ref to a deref place expr is always safe.
                    Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                        self.body
                            .walk_child_exprs_without_pats(expr, |child| self.walk_expr(child));

                        return;
                    }
                    _ => (),
                }
            }
            Expr::MethodCall { .. } => {
                if let Some((func, _)) = self.infer.method_resolution(current) {
                    self.check_call(current, func);
                }
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if let TyKind::Raw(..) = &self.infer[*expr].kind(Interner) {
                    self.on_unsafe_op(current.into(), UnsafetyReason::RawPtrDeref);
                }
            }
            &Expr::Assignment { target, value: _ } => {
                let old_inside_assignment = mem::replace(&mut self.inside_assignment, true);
                self.walk_pats_top(std::iter::once(target), current);
                self.inside_assignment = old_inside_assignment;
            }
            Expr::InlineAsm(asm) => {
                self.on_unsafe_op(current.into(), UnsafetyReason::InlineAsm);
                asm.operands.iter().for_each(|(_, op)| match op {
                    AsmOperand::In { expr, .. }
                    | AsmOperand::Out { expr: Some(expr), .. }
                    | AsmOperand::InOut { expr, .. }
                    | AsmOperand::Const(expr) => self.walk_expr(*expr),
                    AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                        self.walk_expr(*in_expr);
                        if let Some(out_expr) = out_expr {
                            self.walk_expr(*out_expr);
                        }
                    }
                    AsmOperand::Out { expr: None, .. } | AsmOperand::Sym(_) => (),
                    AsmOperand::Label(expr) => {
                        // Inline asm labels are considered safe even when inside unsafe blocks.
                        self.with_inside_unsafe_block(InsideUnsafeBlock::No, |this| {
                            this.walk_expr(*expr)
                        });
                    }
                });
                return;
            }
            // rustc allows union assignment to propagate through field accesses and casts.
            Expr::Cast { .. } => self.inside_assignment = inside_assignment,
            Expr::Field { .. } => {
                self.inside_assignment = inside_assignment;
                if !inside_assignment {
                    if let Some(Either::Left(FieldId { parent: VariantId::UnionId(_), .. })) =
                        self.infer.field_resolution(current)
                    {
                        self.on_unsafe_op(current.into(), UnsafetyReason::UnionField);
                    }
                }
            }
            Expr::Unsafe { statements, .. } => {
                self.with_inside_unsafe_block(InsideUnsafeBlock::Yes, |this| {
                    this.walk_pats_top(
                        statements.iter().filter_map(|statement| match statement {
                            &Statement::Let { pat, .. } => Some(pat),
                            _ => None,
                        }),
                        current,
                    );
                    this.body.walk_child_exprs_without_pats(current, |child| this.walk_expr(child));
                });
                return;
            }
            Expr::Block { statements, .. } | Expr::Async { statements, .. } => {
                self.walk_pats_top(
                    statements.iter().filter_map(|statement| match statement {
                        &Statement::Let { pat, .. } => Some(pat),
                        _ => None,
                    }),
                    current,
                );
            }
            Expr::Match { arms, .. } => {
                self.walk_pats_top(arms.iter().map(|arm| arm.pat), current);
            }
            &Expr::Let { pat, .. } => {
                self.walk_pats_top(std::iter::once(pat), current);
            }
            Expr::Closure { args, .. } => {
                self.walk_pats_top(args.iter().copied(), current);
            }
            Expr::Const(e) => self.walk_expr(*e),
            _ => {}
        }

        self.body.walk_child_exprs_without_pats(current, |child| self.walk_expr(child));
    }

    fn mark_unsafe_path(&mut self, node: ExprOrPatId, path: &Path) {
        let hygiene = self.body.expr_or_pat_path_hygiene(node);
        let value_or_partial = self.resolver.resolve_path_in_value_ns(self.db, path, hygiene);
        if let Some(ResolveValueResult::ValueNs(ValueNs::StaticId(id), _)) = value_or_partial {
            let static_data = self.db.static_signature(id);
            if static_data.flags.contains(StaticFlags::MUTABLE) {
                self.on_unsafe_op(node, UnsafetyReason::MutableStatic);
            } else if static_data.flags.contains(StaticFlags::EXTERN)
                && !static_data.flags.contains(StaticFlags::EXPLICIT_SAFE)
            {
                self.on_unsafe_op(node, UnsafetyReason::ExternStatic);
            }
        }
    }
}
