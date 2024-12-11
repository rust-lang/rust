//! Provides validations for unsafe code. Currently checks if unsafe functions are missing
//! unsafe blocks.

use std::mem;

use either::Either;
use hir_def::{
    body::Body,
    hir::{Expr, ExprId, ExprOrPatId, Pat, PatId, Statement, UnaryOp},
    path::Path,
    resolver::{HasResolver, ResolveValueResult, Resolver, ValueNs},
    type_ref::Rawness,
    AdtId, DefWithBodyId, FieldId, VariantId,
};

use crate::{
    db::HirDatabase, utils::is_fn_unsafe_to_call, InferenceResult, Interner, TyExt, TyKind,
};

/// Returns `(unsafe_exprs, fn_is_unsafe)`.
///
/// If `fn_is_unsafe` is false, `unsafe_exprs` are hard errors. If true, they're `unsafe_op_in_unsafe_fn`.
pub fn missing_unsafe(
    db: &dyn HirDatabase,
    def: DefWithBodyId,
) -> (Vec<(ExprOrPatId, UnsafetyReason)>, bool) {
    let _p = tracing::info_span!("missing_unsafe").entered();

    let mut res = Vec::new();
    let is_unsafe = match def {
        DefWithBodyId::FunctionId(it) => db.function_data(it).is_unsafe(),
        DefWithBodyId::StaticId(_)
        | DefWithBodyId::ConstId(_)
        | DefWithBodyId::VariantId(_)
        | DefWithBodyId::InTypeConstId(_) => false,
    };

    let body = db.body(def);
    let infer = db.infer(def);
    let mut callback = |node, inside_unsafe_block, reason| {
        if inside_unsafe_block == InsideUnsafeBlock::No {
            res.push((node, reason));
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

    (res, is_unsafe)
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

pub fn unsafe_expressions(
    db: &dyn HirDatabase,
    infer: &InferenceResult,
    def: DefWithBodyId,
    body: &Body,
    current: ExprId,
    unsafe_expr_cb: &mut dyn FnMut(ExprOrPatId, InsideUnsafeBlock, UnsafetyReason),
) {
    let mut visitor = UnsafeVisitor::new(db, infer, body, def, unsafe_expr_cb);
    _ = visitor.resolver.update_to_inner_scope(db.upcast(), def, current);
    visitor.walk_expr(current);
}

struct UnsafeVisitor<'a> {
    db: &'a dyn HirDatabase,
    infer: &'a InferenceResult,
    body: &'a Body,
    resolver: Resolver,
    def: DefWithBodyId,
    inside_unsafe_block: InsideUnsafeBlock,
    inside_assignment: bool,
    inside_union_destructure: bool,
    unsafe_expr_cb: &'a mut dyn FnMut(ExprOrPatId, InsideUnsafeBlock, UnsafetyReason),
}

impl<'a> UnsafeVisitor<'a> {
    fn new(
        db: &'a dyn HirDatabase,
        infer: &'a InferenceResult,
        body: &'a Body,
        def: DefWithBodyId,
        unsafe_expr_cb: &'a mut dyn FnMut(ExprOrPatId, InsideUnsafeBlock, UnsafetyReason),
    ) -> Self {
        let resolver = def.resolver(db.upcast());
        Self {
            db,
            infer,
            body,
            resolver,
            def,
            inside_unsafe_block: InsideUnsafeBlock::No,
            inside_assignment: false,
            inside_union_destructure: false,
            unsafe_expr_cb,
        }
    }

    fn call_cb(&mut self, node: ExprOrPatId, reason: UnsafetyReason) {
        (self.unsafe_expr_cb)(node, self.inside_unsafe_block, reason);
    }

    fn walk_pats_top(&mut self, pats: impl Iterator<Item = PatId>, parent_expr: ExprId) {
        let guard = self.resolver.update_to_inner_scope(self.db.upcast(), self.def, parent_expr);
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
                | Pat::ConstBlock(..) => self.call_cb(current.into(), UnsafetyReason::UnionField),
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
                if let Some(func) = self.infer[callee].as_fn_def(self.db) {
                    if is_fn_unsafe_to_call(self.db, func) {
                        self.call_cb(current.into(), UnsafetyReason::UnsafeFnCall);
                    }
                }
            }
            Expr::Path(path) => {
                let guard =
                    self.resolver.update_to_inner_scope(self.db.upcast(), self.def, current);
                self.mark_unsafe_path(current.into(), path);
                self.resolver.reset_to_guard(guard);
            }
            Expr::Ref { expr, rawness: Rawness::RawPtr, mutability: _ } => {
                if let Expr::Path(_) = self.body.exprs[*expr] {
                    // Do not report unsafe for `addr_of[_mut]!(EXTERN_OR_MUT_STATIC)`,
                    // see https://github.com/rust-lang/rust/pull/125834.
                    return;
                }
            }
            Expr::MethodCall { .. } => {
                if self
                    .infer
                    .method_resolution(current)
                    .map(|(func, _)| is_fn_unsafe_to_call(self.db, func))
                    .unwrap_or(false)
                {
                    self.call_cb(current.into(), UnsafetyReason::UnsafeFnCall);
                }
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                if let TyKind::Raw(..) = &self.infer[*expr].kind(Interner) {
                    self.call_cb(current.into(), UnsafetyReason::RawPtrDeref);
                }
            }
            Expr::Unsafe { .. } => {
                let old_inside_unsafe_block =
                    mem::replace(&mut self.inside_unsafe_block, InsideUnsafeBlock::Yes);
                self.body.walk_child_exprs_without_pats(current, |child| self.walk_expr(child));
                self.inside_unsafe_block = old_inside_unsafe_block;
                return;
            }
            &Expr::Assignment { target, value: _ } => {
                let old_inside_assignment = mem::replace(&mut self.inside_assignment, true);
                self.walk_pats_top(std::iter::once(target), current);
                self.inside_assignment = old_inside_assignment;
            }
            Expr::InlineAsm(_) => self.call_cb(current.into(), UnsafetyReason::InlineAsm),
            // rustc allows union assignment to propagate through field accesses and casts.
            Expr::Cast { .. } => self.inside_assignment = inside_assignment,
            Expr::Field { .. } => {
                self.inside_assignment = inside_assignment;
                if !inside_assignment {
                    if let Some(Either::Left(FieldId { parent: VariantId::UnionId(_), .. })) =
                        self.infer.field_resolution(current)
                    {
                        self.call_cb(current.into(), UnsafetyReason::UnionField);
                    }
                }
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
            _ => {}
        }

        self.body.walk_child_exprs_without_pats(current, |child| self.walk_expr(child));
    }

    fn mark_unsafe_path(&mut self, node: ExprOrPatId, path: &Path) {
        let hygiene = self.body.expr_or_pat_path_hygiene(node);
        let value_or_partial =
            self.resolver.resolve_path_in_value_ns(self.db.upcast(), path, hygiene);
        if let Some(ResolveValueResult::ValueNs(ValueNs::StaticId(id), _)) = value_or_partial {
            let static_data = self.db.static_data(id);
            if static_data.mutable {
                self.call_cb(node, UnsafetyReason::MutableStatic);
            } else if static_data.is_extern && !static_data.has_safe_kw {
                self.call_cb(node, UnsafetyReason::ExternStatic);
            }
        }
    }
}
