//! Finds if an expression is an immutable context or a mutable context, which is used in selecting
//! between `Deref` and `DerefMut` or `Index` and `IndexMut` or similar.

use chalk_ir::{Mutability, cast::Cast};
use hir_def::{
    hir::{
        Array, AsmOperand, BinaryOp, BindingAnnotation, Expr, ExprId, Pat, PatId, Statement,
        UnaryOp,
    },
    lang_item::LangItem,
};
use hir_expand::name::Name;
use intern::sym;

use crate::{
    Adjust, Adjustment, AutoBorrow, Interner, OverloadedDeref, TyBuilder, TyKind,
    infer::{Expectation, InferenceContext, expr::ExprIsRead},
    lower::lower_to_chalk_mutability,
};

impl InferenceContext<'_> {
    pub(crate) fn infer_mut_body(&mut self) {
        self.infer_mut_expr(self.body.body_expr, Mutability::Not);
    }

    fn infer_mut_expr(&mut self, tgt_expr: ExprId, mut mutability: Mutability) {
        if let Some(adjustments) = self.result.expr_adjustments.get_mut(&tgt_expr) {
            for adj in adjustments.iter_mut().rev() {
                match &mut adj.kind {
                    Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => (),
                    Adjust::Deref(Some(d)) => *d = OverloadedDeref(Some(mutability)),
                    Adjust::Borrow(b) => match b {
                        AutoBorrow::Ref(_, m) | AutoBorrow::RawPtr(m) => mutability = *m,
                    },
                }
            }
        }
        self.infer_mut_expr_without_adjust(tgt_expr, mutability);
    }

    fn infer_mut_expr_without_adjust(&mut self, tgt_expr: ExprId, mutability: Mutability) {
        match &self.body[tgt_expr] {
            Expr::Missing => (),
            Expr::InlineAsm(e) => {
                e.operands.iter().for_each(|(_, op)| match op {
                    AsmOperand::In { expr, .. }
                    | AsmOperand::Out { expr: Some(expr), .. }
                    | AsmOperand::InOut { expr, .. } => {
                        self.infer_mut_expr_without_adjust(*expr, Mutability::Not)
                    }
                    AsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                        self.infer_mut_expr_without_adjust(*in_expr, Mutability::Not);
                        if let Some(out_expr) = out_expr {
                            self.infer_mut_expr_without_adjust(*out_expr, Mutability::Not);
                        }
                    }
                    AsmOperand::Out { expr: None, .. }
                    | AsmOperand::Label(_)
                    | AsmOperand::Sym(_)
                    | AsmOperand::Const(_) => (),
                });
            }
            Expr::OffsetOf(_) => (),
            &Expr::If { condition, then_branch, else_branch } => {
                self.infer_mut_expr(condition, Mutability::Not);
                self.infer_mut_expr(then_branch, Mutability::Not);
                if let Some(else_branch) = else_branch {
                    self.infer_mut_expr(else_branch, Mutability::Not);
                }
            }
            Expr::Const(id) => {
                self.infer_mut_expr(*id, Mutability::Not);
            }
            Expr::Let { pat, expr } => self.infer_mut_expr(*expr, self.pat_bound_mutability(*pat)),
            Expr::Block { id: _, statements, tail, label: _ }
            | Expr::Async { id: _, statements, tail }
            | Expr::Unsafe { id: _, statements, tail } => {
                for st in statements.iter() {
                    match st {
                        Statement::Let { pat, type_ref: _, initializer, else_branch } => {
                            if let Some(i) = initializer {
                                self.infer_mut_expr(*i, self.pat_bound_mutability(*pat));
                            }
                            if let Some(e) = else_branch {
                                self.infer_mut_expr(*e, Mutability::Not);
                            }
                        }
                        Statement::Expr { expr, has_semi: _ } => {
                            self.infer_mut_expr(*expr, Mutability::Not);
                        }
                        Statement::Item(_) => (),
                    }
                }
                if let Some(tail) = tail {
                    self.infer_mut_expr(*tail, Mutability::Not);
                }
            }
            Expr::MethodCall { receiver: it, method_name: _, args, generic_args: _ }
            | Expr::Call { callee: it, args } => {
                self.infer_mut_not_expr_iter(args.iter().copied().chain(Some(*it)));
            }
            Expr::Match { expr, arms } => {
                let m = self.pat_iter_bound_mutability(arms.iter().map(|it| it.pat));
                self.infer_mut_expr(*expr, m);
                for arm in arms.iter() {
                    self.infer_mut_expr(arm.expr, Mutability::Not);
                    if let Some(g) = arm.guard {
                        self.infer_mut_expr(g, Mutability::Not);
                    }
                }
            }
            Expr::Yield { expr }
            | Expr::Yeet { expr }
            | Expr::Return { expr }
            | Expr::Break { expr, label: _ } => {
                if let &Some(expr) = expr {
                    self.infer_mut_expr(expr, Mutability::Not);
                }
            }
            Expr::Become { expr } => {
                self.infer_mut_expr(*expr, Mutability::Not);
            }
            Expr::RecordLit { path: _, fields, spread } => {
                self.infer_mut_not_expr_iter(fields.iter().map(|it| it.expr).chain(*spread))
            }
            &Expr::Index { base, index } => {
                if mutability == Mutability::Mut {
                    if let Some((f, _)) = self.result.method_resolutions.get_mut(&tgt_expr) {
                        if let Some(index_trait) =
                            LangItem::IndexMut.resolve_trait(self.db, self.table.trait_env.krate)
                        {
                            if let Some(index_fn) = index_trait
                                .trait_items(self.db)
                                .method_by_name(&Name::new_symbol_root(sym::index_mut))
                            {
                                *f = index_fn;
                                let mut base_ty = None;
                                let base_adjustments = self
                                    .result
                                    .expr_adjustments
                                    .get_mut(&base)
                                    .and_then(|it| it.last_mut());
                                if let Some(Adjustment {
                                    kind: Adjust::Borrow(AutoBorrow::Ref(_, mutability)),
                                    target,
                                }) = base_adjustments
                                {
                                    if let TyKind::Ref(_, _, ty) = target.kind(Interner) {
                                        base_ty = Some(ty.clone());
                                    }
                                    *mutability = Mutability::Mut;
                                }

                                // Apply `IndexMut` obligation for non-assignee expr
                                if let Some(base_ty) = base_ty {
                                    let index_ty =
                                        if let Some(ty) = self.result.type_of_expr.get(index) {
                                            ty.clone()
                                        } else {
                                            self.infer_expr(
                                                index,
                                                &Expectation::none(),
                                                ExprIsRead::Yes,
                                            )
                                        };
                                    let trait_ref = TyBuilder::trait_ref(self.db, index_trait)
                                        .push(base_ty)
                                        .fill(|_| index_ty.clone().cast(Interner))
                                        .build();
                                    self.push_obligation(trait_ref.cast(Interner));
                                }
                            }
                        }
                    }
                }
                self.infer_mut_expr(base, mutability);
                self.infer_mut_expr(index, Mutability::Not);
            }
            Expr::UnaryOp { expr, op: UnaryOp::Deref } => {
                let mut mutability = mutability;
                if let Some((f, _)) = self.result.method_resolutions.get_mut(&tgt_expr) {
                    if mutability == Mutability::Mut {
                        if let Some(deref_trait) =
                            LangItem::DerefMut.resolve_trait(self.db, self.table.trait_env.krate)
                        {
                            let ty = self.result.type_of_expr.get(*expr);
                            let is_mut_ptr = ty.is_some_and(|ty| {
                                let ty = self.table.resolve_ty_shallow(ty);
                                matches!(
                                    ty.kind(Interner),
                                    chalk_ir::TyKind::Raw(Mutability::Mut, _)
                                )
                            });
                            if is_mut_ptr {
                                mutability = Mutability::Not;
                            } else if let Some(deref_fn) = deref_trait
                                .trait_items(self.db)
                                .method_by_name(&Name::new_symbol_root(sym::deref_mut))
                            {
                                *f = deref_fn;
                            }
                        }
                    }
                }
                self.infer_mut_expr(*expr, mutability);
            }
            Expr::Field { expr, name: _ } => {
                self.infer_mut_expr(*expr, mutability);
            }
            Expr::UnaryOp { expr, op: _ }
            | Expr::Range { lhs: Some(expr), rhs: None, range_type: _ }
            | Expr::Range { rhs: Some(expr), lhs: None, range_type: _ }
            | Expr::Await { expr }
            | Expr::Box { expr }
            | Expr::Loop { body: expr, label: _ }
            | Expr::Cast { expr, type_ref: _ } => {
                self.infer_mut_expr(*expr, Mutability::Not);
            }
            Expr::Ref { expr, rawness: _, mutability } => {
                let mutability = lower_to_chalk_mutability(*mutability);
                self.infer_mut_expr(*expr, mutability);
            }
            Expr::BinaryOp { lhs, rhs, op: Some(BinaryOp::Assignment { .. }) } => {
                self.infer_mut_expr(*lhs, Mutability::Mut);
                self.infer_mut_expr(*rhs, Mutability::Not);
            }
            &Expr::Assignment { target, value } => {
                self.body.walk_pats(target, &mut |pat| match self.body[pat] {
                    Pat::Expr(expr) => self.infer_mut_expr(expr, Mutability::Mut),
                    Pat::ConstBlock(block) => self.infer_mut_expr(block, Mutability::Not),
                    _ => {}
                });
                self.infer_mut_expr(value, Mutability::Not);
            }
            Expr::Array(Array::Repeat { initializer: lhs, repeat: rhs })
            | Expr::BinaryOp { lhs, rhs, op: _ }
            | Expr::Range { lhs: Some(lhs), rhs: Some(rhs), range_type: _ } => {
                self.infer_mut_expr(*lhs, Mutability::Not);
                self.infer_mut_expr(*rhs, Mutability::Not);
            }
            Expr::Closure { body, .. } => {
                self.infer_mut_expr(*body, Mutability::Not);
            }
            Expr::Tuple { exprs } | Expr::Array(Array::ElementList { elements: exprs }) => {
                self.infer_mut_not_expr_iter(exprs.iter().copied());
            }
            // These don't need any action, as they don't have sub expressions
            Expr::Range { lhs: None, rhs: None, range_type: _ }
            | Expr::Literal(_)
            | Expr::Path(_)
            | Expr::Continue { .. }
            | Expr::Underscore => (),
        }
    }

    fn infer_mut_not_expr_iter(&mut self, exprs: impl Iterator<Item = ExprId>) {
        for expr in exprs {
            self.infer_mut_expr(expr, Mutability::Not);
        }
    }

    fn pat_iter_bound_mutability(&self, mut pat: impl Iterator<Item = PatId>) -> Mutability {
        if pat.any(|p| self.pat_bound_mutability(p) == Mutability::Mut) {
            Mutability::Mut
        } else {
            Mutability::Not
        }
    }

    /// Checks if the pat contains a `ref mut` binding. Such paths makes the context of bounded expressions
    /// mutable. For example in `let (ref mut x0, ref x1) = *it;` we need to use `DerefMut` for `*it` but in
    /// `let (ref x0, ref x1) = *it;` we should use `Deref`.
    fn pat_bound_mutability(&self, pat: PatId) -> Mutability {
        let mut r = Mutability::Not;
        self.body.walk_bindings_in_pat(pat, |b| {
            if self.body.bindings[b].mode == BindingAnnotation::RefMut {
                r = Mutability::Mut;
            }
        });
        r
    }
}
