use crate::Lint;
use crate::{EarlyContext, EarlyLintPass, LateContext, LateLintPass, LintContext};
use rustc_ast::ast;
use rustc_ast::ast::{ExprKind, StmtKind};
use rustc_ast::attr;
use rustc_ast::util::parser;
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashMap;
use rustc_errors::{pluralize, Applicability};
use rustc_feature::{AttributeType, BuiltinAttribute, BUILTIN_ATTRIBUTE_MAP};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_middle::ty::adjustment;
use rustc_middle::ty::{self, Ty};
use rustc_session::lint::builtin::UNUSED_ATTRIBUTES;
use rustc_span::symbol::Symbol;
use rustc_span::symbol::{kw, sym};
use rustc_span::{BytePos, Span};

use log::debug;

declare_lint! {
    pub UNUSED_MUST_USE,
    Warn,
    "unused result of a type flagged as `#[must_use]`",
    report_in_external_macro
}

declare_lint! {
    pub UNUSED_RESULTS,
    Allow,
    "unused result of an expression in a statement"
}

declare_lint_pass!(UnusedResults => [UNUSED_MUST_USE, UNUSED_RESULTS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedResults {
    fn check_stmt(&mut self, cx: &LateContext<'_, '_>, s: &hir::Stmt<'_>) {
        let expr = match s.kind {
            hir::StmtKind::Semi(ref expr) => &**expr,
            _ => return,
        };

        if let hir::ExprKind::Ret(..) = expr.kind {
            return;
        }

        let ty = cx.tables.expr_ty(&expr);
        let type_permits_lack_of_use = check_must_use_ty(cx, ty, &expr, s.span, "", "", 1);

        let mut fn_warned = false;
        let mut op_warned = false;
        let maybe_def_id = match expr.kind {
            hir::ExprKind::Call(ref callee, _) => {
                match callee.kind {
                    hir::ExprKind::Path(ref qpath) => {
                        match cx.tables.qpath_res(qpath, callee.hir_id) {
                            Res::Def(DefKind::Fn, def_id) | Res::Def(DefKind::AssocFn, def_id) => {
                                Some(def_id)
                            }
                            // `Res::Local` if it was a closure, for which we
                            // do not currently support must-use linting
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
            hir::ExprKind::MethodCall(..) => cx.tables.type_dependent_def_id(expr.hir_id),
            _ => None,
        };
        if let Some(def_id) = maybe_def_id {
            fn_warned = check_must_use_def(cx, def_id, s.span, "return value of ", "");
        } else if type_permits_lack_of_use {
            // We don't warn about unused unit or uninhabited types.
            // (See https://github.com/rust-lang/rust/issues/43806 for details.)
            return;
        }

        let must_use_op = match expr.kind {
            // Hardcoding operators here seemed more expedient than the
            // refactoring that would be needed to look up the `#[must_use]`
            // attribute which does exist on the comparison trait methods
            hir::ExprKind::Binary(bin_op, ..) => match bin_op.node {
                hir::BinOpKind::Eq
                | hir::BinOpKind::Lt
                | hir::BinOpKind::Le
                | hir::BinOpKind::Ne
                | hir::BinOpKind::Ge
                | hir::BinOpKind::Gt => Some("comparison"),
                hir::BinOpKind::Add
                | hir::BinOpKind::Sub
                | hir::BinOpKind::Div
                | hir::BinOpKind::Mul
                | hir::BinOpKind::Rem => Some("arithmetic operation"),
                hir::BinOpKind::And | hir::BinOpKind::Or => Some("logical operation"),
                hir::BinOpKind::BitXor
                | hir::BinOpKind::BitAnd
                | hir::BinOpKind::BitOr
                | hir::BinOpKind::Shl
                | hir::BinOpKind::Shr => Some("bitwise operation"),
            },
            hir::ExprKind::Unary(..) => Some("unary operation"),
            _ => None,
        };

        if let Some(must_use_op) = must_use_op {
            cx.struct_span_lint(UNUSED_MUST_USE, expr.span, |lint| {
                lint.build(&format!("unused {} that must be used", must_use_op)).emit()
            });
            op_warned = true;
        }

        if !(type_permits_lack_of_use || fn_warned || op_warned) {
            cx.struct_span_lint(UNUSED_RESULTS, s.span, |lint| lint.build("unused result").emit());
        }

        // Returns whether an error has been emitted (and thus another does not need to be later).
        fn check_must_use_ty<'tcx>(
            cx: &LateContext<'_, 'tcx>,
            ty: Ty<'tcx>,
            expr: &hir::Expr<'_>,
            span: Span,
            descr_pre: &str,
            descr_post: &str,
            plural_len: usize,
        ) -> bool {
            if ty.is_unit()
                || cx.tcx.is_ty_uninhabited_from(
                    cx.tcx.parent_module(expr.hir_id).to_def_id(),
                    ty,
                    cx.param_env,
                )
            {
                return true;
            }

            let plural_suffix = pluralize!(plural_len);

            match ty.kind {
                ty::Adt(..) if ty.is_box() => {
                    let boxed_ty = ty.boxed_ty();
                    let descr_pre = &format!("{}boxed ", descr_pre);
                    check_must_use_ty(cx, boxed_ty, expr, span, descr_pre, descr_post, plural_len)
                }
                ty::Adt(def, _) => check_must_use_def(cx, def.did, span, descr_pre, descr_post),
                ty::Opaque(def, _) => {
                    let mut has_emitted = false;
                    for (predicate, _) in cx.tcx.predicates_of(def).predicates {
                        if let ty::Predicate::Trait(ref poly_trait_predicate, _) = predicate {
                            let trait_ref = poly_trait_predicate.skip_binder().trait_ref;
                            let def_id = trait_ref.def_id;
                            let descr_pre =
                                &format!("{}implementer{} of ", descr_pre, plural_suffix,);
                            if check_must_use_def(cx, def_id, span, descr_pre, descr_post) {
                                has_emitted = true;
                                break;
                            }
                        }
                    }
                    has_emitted
                }
                ty::Dynamic(binder, _) => {
                    let mut has_emitted = false;
                    for predicate in binder.skip_binder().iter() {
                        if let ty::ExistentialPredicate::Trait(ref trait_ref) = predicate {
                            let def_id = trait_ref.def_id;
                            let descr_post =
                                &format!(" trait object{}{}", plural_suffix, descr_post,);
                            if check_must_use_def(cx, def_id, span, descr_pre, descr_post) {
                                has_emitted = true;
                                break;
                            }
                        }
                    }
                    has_emitted
                }
                ty::Tuple(ref tys) => {
                    let mut has_emitted = false;
                    let spans = if let hir::ExprKind::Tup(comps) = &expr.kind {
                        debug_assert_eq!(comps.len(), tys.len());
                        comps.iter().map(|e| e.span).collect()
                    } else {
                        vec![]
                    };
                    for (i, ty) in tys.iter().map(|k| k.expect_ty()).enumerate() {
                        let descr_post = &format!(" in tuple element {}", i);
                        let span = *spans.get(i).unwrap_or(&span);
                        if check_must_use_ty(cx, ty, expr, span, descr_pre, descr_post, plural_len)
                        {
                            has_emitted = true;
                        }
                    }
                    has_emitted
                }
                ty::Array(ty, len) => match len.try_eval_usize(cx.tcx, cx.param_env) {
                    // If the array is definitely non-empty, we can do `#[must_use]` checking.
                    Some(n) if n != 0 => {
                        let descr_pre = &format!("{}array{} of ", descr_pre, plural_suffix,);
                        check_must_use_ty(cx, ty, expr, span, descr_pre, descr_post, n as usize + 1)
                    }
                    // Otherwise, we don't lint, to avoid false positives.
                    _ => false,
                },
                _ => false,
            }
        }

        // Returns whether an error has been emitted (and thus another does not need to be later).
        // FIXME: Args desc_{pre,post}_path could be made lazy by taking Fn() -> &str, but this
        // would make calling it a big awkward. Could also take String (so args are moved), but
        // this would still require a copy into the format string, which would only be executed
        // when needed.
        fn check_must_use_def(
            cx: &LateContext<'_, '_>,
            def_id: DefId,
            span: Span,
            descr_pre_path: &str,
            descr_post_path: &str,
        ) -> bool {
            for attr in cx.tcx.get_attrs(def_id).iter() {
                if attr.check_name(sym::must_use) {
                    cx.struct_span_lint(UNUSED_MUST_USE, span, |lint| {
                        let msg = format!(
                            "unused {}`{}`{} that must be used",
                            descr_pre_path,
                            cx.tcx.def_path_str(def_id),
                            descr_post_path
                        );
                        let mut err = lint.build(&msg);
                        // check for #[must_use = "..."]
                        if let Some(note) = attr.value_str() {
                            err.note(&note.as_str());
                        }
                        err.emit();
                    });
                    return true;
                }
            }
            false
        }
    }
}

declare_lint! {
    pub PATH_STATEMENTS,
    Warn,
    "path statements with no effect"
}

declare_lint_pass!(PathStatements => [PATH_STATEMENTS]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for PathStatements {
    fn check_stmt(&mut self, cx: &LateContext<'_, '_>, s: &hir::Stmt<'_>) {
        if let hir::StmtKind::Semi(ref expr) = s.kind {
            if let hir::ExprKind::Path(_) = expr.kind {
                cx.struct_span_lint(PATH_STATEMENTS, s.span, |lint| {
                    lint.build("path statement with no effect").emit()
                });
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct UnusedAttributes {
    builtin_attributes: &'static FxHashMap<Symbol, &'static BuiltinAttribute>,
}

impl UnusedAttributes {
    pub fn new() -> Self {
        UnusedAttributes { builtin_attributes: &*BUILTIN_ATTRIBUTE_MAP }
    }
}

impl_lint_pass!(UnusedAttributes => [UNUSED_ATTRIBUTES]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedAttributes {
    fn check_attribute(&mut self, cx: &LateContext<'_, '_>, attr: &ast::Attribute) {
        debug!("checking attribute: {:?}", attr);

        if attr.is_doc_comment() {
            return;
        }

        let attr_info = attr.ident().and_then(|ident| self.builtin_attributes.get(&ident.name));

        if let Some(&&(name, ty, ..)) = attr_info {
            if let AttributeType::Whitelisted = ty {
                debug!("{:?} is Whitelisted", name);
                return;
            }
        }

        if !attr::is_used(attr) {
            debug!("emitting warning for: {:?}", attr);
            cx.struct_span_lint(UNUSED_ATTRIBUTES, attr.span, |lint| {
                lint.build("unused attribute").emit()
            });
            // Is it a builtin attribute that must be used at the crate level?
            if attr_info.map_or(false, |(_, ty, ..)| ty == &AttributeType::CrateLevel) {
                cx.struct_span_lint(UNUSED_ATTRIBUTES, attr.span, |lint| {
                    let msg = match attr.style {
                        ast::AttrStyle::Outer => {
                            "crate-level attribute should be an inner attribute: add an exclamation \
                             mark: `#![foo]`"
                        }
                        ast::AttrStyle::Inner => "crate-level attribute should be in the root module",
                    };
                    lint.build(msg).emit()
                });
            }
        } else {
            debug!("Attr was used: {:?}", attr);
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum UnusedDelimsCtx {
    FunctionArg,
    MethodArg,
    AssignedValue,
    IfCond,
    WhileCond,
    ForIterExpr,
    MatchScrutineeExpr,
    ReturnValue,
    BlockRetValue,
    LetScrutineeExpr,
    ArrayLenExpr,
    AnonConst,
}

impl From<UnusedDelimsCtx> for &'static str {
    fn from(ctx: UnusedDelimsCtx) -> &'static str {
        match ctx {
            UnusedDelimsCtx::FunctionArg => "function argument",
            UnusedDelimsCtx::MethodArg => "method argument",
            UnusedDelimsCtx::AssignedValue => "assigned value",
            UnusedDelimsCtx::IfCond => "`if` condition",
            UnusedDelimsCtx::WhileCond => "`while` condition",
            UnusedDelimsCtx::ForIterExpr => "`for` iterator expression",
            UnusedDelimsCtx::MatchScrutineeExpr => "`match` scrutinee expression",
            UnusedDelimsCtx::ReturnValue => "`return` value",
            UnusedDelimsCtx::BlockRetValue => "block return value",
            UnusedDelimsCtx::LetScrutineeExpr => "`let` scrutinee expression",
            UnusedDelimsCtx::ArrayLenExpr | UnusedDelimsCtx::AnonConst => "const expression",
        }
    }
}

/// Used by both `UnusedParens` and `UnusedBraces` to prevent code duplication.
trait UnusedDelimLint {
    const DELIM_STR: &'static str;

    /// Due to `ref` pattern, there can be a difference between using
    /// `{ expr }` and `expr` in pattern-matching contexts. This means
    /// that we should only lint `unused_parens` and not `unused_braces`
    /// in this case.
    ///
    /// ```rust
    /// let mut a = 7;
    /// let ref b = { a }; // We actually borrow a copy of `a` here.
    /// a += 1; // By mutating `a` we invalidate any borrows of `a`.
    /// assert_eq!(b + 1, a); // `b` does not borrow `a`, so we can still use it here.
    /// ```
    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool;

    // this cannot be a constant is it refers to a static.
    fn lint(&self) -> &'static Lint;

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
    );

    fn is_expr_delims_necessary(inner: &ast::Expr, followed_by_block: bool) -> bool {
        followed_by_block
            && match inner.kind {
                ExprKind::Ret(_) | ExprKind::Break(..) => true,
                _ => parser::contains_exterior_struct_lit(&inner),
            }
    }

    fn emit_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
    ) {
        let expr_text = if let Ok(snippet) = cx.sess().source_map().span_to_snippet(value.span) {
            snippet
        } else {
            pprust::expr_to_string(value)
        };
        let keep_space = (
            left_pos.map(|s| s >= value.span.lo()).unwrap_or(false),
            right_pos.map(|s| s <= value.span.hi()).unwrap_or(false),
        );
        self.emit_unused_delims(cx, value.span, &expr_text, ctx.into(), keep_space);
    }

    fn emit_unused_delims(
        &self,
        cx: &EarlyContext<'_>,
        span: Span,
        pattern: &str,
        msg: &str,
        keep_space: (bool, bool),
    ) {
        cx.struct_span_lint(self.lint(), span, |lint| {
            let span_msg = format!("unnecessary {} around {}", Self::DELIM_STR, msg);
            let mut err = lint.build(&span_msg);
            let mut ate_left_paren = false;
            let mut ate_right_paren = false;
            let parens_removed = pattern.trim_matches(|c| match c {
                '(' | '{' => {
                    if ate_left_paren {
                        false
                    } else {
                        ate_left_paren = true;
                        true
                    }
                }
                ')' | '}' => {
                    if ate_right_paren {
                        false
                    } else {
                        ate_right_paren = true;
                        true
                    }
                }
                _ => false,
            });

            let replace = {
                let mut replace = if keep_space.0 {
                    let mut s = String::from(" ");
                    s.push_str(parens_removed);
                    s
                } else {
                    String::from(parens_removed)
                };

                if keep_space.1 {
                    replace.push(' ');
                }
                replace
            };

            let suggestion = format!("remove these {}", Self::DELIM_STR);

            err.span_suggestion_short(span, &suggestion, replace, Applicability::MachineApplicable);
            err.emit();
        });
    }

    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        use rustc_ast::ast::ExprKind::*;
        let (value, ctx, followed_by_block, left_pos, right_pos) = match e.kind {
            // Do not lint `unused_braces` in `if let` expressions.
            If(ref cond, ref block, ..)
                if !matches!(cond.kind, Let(_, _)) || Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX =>
            {
                let left = e.span.lo() + rustc_span::BytePos(2);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::IfCond, true, Some(left), Some(right))
            }

            While(ref cond, ref block, ..) => {
                let left = e.span.lo() + rustc_span::BytePos(5);
                let right = block.span.lo();
                (cond, UnusedDelimsCtx::WhileCond, true, Some(left), Some(right))
            }

            ForLoop(_, ref cond, ref block, ..) => {
                (cond, UnusedDelimsCtx::ForIterExpr, true, None, Some(block.span.lo()))
            }

            Match(ref head, _) if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX => {
                let left = e.span.lo() + rustc_span::BytePos(5);
                (head, UnusedDelimsCtx::MatchScrutineeExpr, true, Some(left), None)
            }

            Ret(Some(ref value)) => {
                let left = e.span.lo() + rustc_span::BytePos(3);
                (value, UnusedDelimsCtx::ReturnValue, false, Some(left), None)
            }

            Assign(_, ref value, _) | AssignOp(.., ref value) => {
                (value, UnusedDelimsCtx::AssignedValue, false, None, None)
            }
            // either function/method call, or something this lint doesn't care about
            ref call_or_other => {
                let (args_to_check, ctx) = match *call_or_other {
                    Call(_, ref args) => (&args[..], UnusedDelimsCtx::FunctionArg),
                    // first "argument" is self (which sometimes needs delims)
                    MethodCall(_, ref args) => (&args[1..], UnusedDelimsCtx::MethodArg),
                    // actual catch-all arm
                    _ => {
                        return;
                    }
                };
                // Don't lint if this is a nested macro expansion: otherwise, the lint could
                // trigger in situations that macro authors shouldn't have to care about, e.g.,
                // when a parenthesized token tree matched in one macro expansion is matched as
                // an expression in another and used as a fn/method argument (Issue #47775)
                if e.span.ctxt().outer_expn_data().call_site.from_expansion() {
                    return;
                }
                for arg in args_to_check {
                    self.check_unused_delims_expr(cx, arg, ctx, false, None, None);
                }
                return;
            }
        };
        self.check_unused_delims_expr(cx, &value, ctx, followed_by_block, left_pos, right_pos);
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        match s.kind {
            StmtKind::Local(ref local) if Self::LINT_EXPR_IN_PATTERN_MATCHING_CTX => {
                if let Some(ref value) = local.init {
                    self.check_unused_delims_expr(
                        cx,
                        &value,
                        UnusedDelimsCtx::AssignedValue,
                        false,
                        None,
                        None,
                    );
                }
            }
            StmtKind::Expr(ref expr) => {
                self.check_unused_delims_expr(
                    cx,
                    &expr,
                    UnusedDelimsCtx::BlockRetValue,
                    false,
                    None,
                    None,
                );
            }
            _ => {}
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        use ast::ItemKind::*;

        if let Const(.., Some(expr)) | Static(.., Some(expr)) = &item.kind {
            self.check_unused_delims_expr(
                cx,
                expr,
                UnusedDelimsCtx::AssignedValue,
                false,
                None,
                None,
            );
        }
    }
}

declare_lint! {
    pub(super) UNUSED_PARENS,
    Warn,
    "`if`, `match`, `while` and `return` do not need parentheses"
}

declare_lint_pass!(UnusedParens => [UNUSED_PARENS]);

impl UnusedDelimLint for UnusedParens {
    const DELIM_STR: &'static str = "parentheses";

    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool = true;

    fn lint(&self) -> &'static Lint {
        UNUSED_PARENS
    }

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
    ) {
        match value.kind {
            ast::ExprKind::Paren(ref inner) => {
                if !Self::is_expr_delims_necessary(inner, followed_by_block)
                    && value.attrs.is_empty()
                    && !value.span.from_expansion()
                {
                    self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos)
                }
            }
            ast::ExprKind::Let(_, ref expr) => {
                // FIXME(#60336): Properly handle `let true = (false && true)`
                // actually needing the parenthesis.
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
                );
            }
            _ => {}
        }
    }
}

impl UnusedParens {
    fn check_unused_parens_pat(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Pat,
        avoid_or: bool,
        avoid_mut: bool,
    ) {
        use ast::{BindingMode, Mutability, PatKind};

        if let PatKind::Paren(inner) = &value.kind {
            match inner.kind {
                // The lint visitor will visit each subpattern of `p`. We do not want to lint
                // any range pattern no matter where it occurs in the pattern. For something like
                // `&(a..=b)`, there is a recursive `check_pat` on `a` and `b`, but we will assume
                // that if there are unnecessary parens they serve a purpose of readability.
                PatKind::Range(..) => return,
                // Avoid `p0 | .. | pn` if we should.
                PatKind::Or(..) if avoid_or => return,
                // Avoid `mut x` and `mut x @ p` if we should:
                PatKind::Ident(BindingMode::ByValue(Mutability::Mut), ..) if avoid_mut => return,
                // Otherwise proceed with linting.
                _ => {}
            }

            let pattern_text =
                if let Ok(snippet) = cx.sess().source_map().span_to_snippet(value.span) {
                    snippet
                } else {
                    pprust::pat_to_string(value)
                };
            self.emit_unused_delims(cx, value.span, &pattern_text, "pattern", (false, false));
        }
    }
}

impl EarlyLintPass for UnusedParens {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        if let ExprKind::Let(ref pat, ..) | ExprKind::ForLoop(ref pat, ..) = e.kind {
            self.check_unused_parens_pat(cx, pat, false, false);
        }

        <Self as UnusedDelimLint>::check_expr(self, cx, e)
    }

    fn check_pat(&mut self, cx: &EarlyContext<'_>, p: &ast::Pat) {
        use ast::{Mutability, PatKind::*};
        match &p.kind {
            // Do not lint on `(..)` as that will result in the other arms being useless.
            Paren(_)
            // The other cases do not contain sub-patterns.
            | Wild | Rest | Lit(..) | MacCall(..) | Range(..) | Ident(.., None) | Path(..) => {},
            // These are list-like patterns; parens can always be removed.
            TupleStruct(_, ps) | Tuple(ps) | Slice(ps) | Or(ps) => for p in ps {
                self.check_unused_parens_pat(cx, p, false, false);
            },
            Struct(_, fps, _) => for f in fps {
                self.check_unused_parens_pat(cx, &f.pat, false, false);
            },
            // Avoid linting on `i @ (p0 | .. | pn)` and `box (p0 | .. | pn)`, #64106.
            Ident(.., Some(p)) | Box(p) => self.check_unused_parens_pat(cx, p, true, false),
            // Avoid linting on `&(mut x)` as `&mut x` has a different meaning, #55342.
            // Also avoid linting on `& mut? (p0 | .. | pn)`, #64106.
            Ref(p, m) => self.check_unused_parens_pat(cx, p, true, *m == Mutability::Not),
        }
    }

    fn check_anon_const(&mut self, cx: &EarlyContext<'_>, c: &ast::AnonConst) {
        self.check_unused_delims_expr(cx, &c.value, UnusedDelimsCtx::AnonConst, false, None, None);
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        if let StmtKind::Local(ref local) = s.kind {
            self.check_unused_parens_pat(cx, &local.pat, false, false);
        }

        <Self as UnusedDelimLint>::check_stmt(self, cx, s)
    }

    fn check_param(&mut self, cx: &EarlyContext<'_>, param: &ast::Param) {
        self.check_unused_parens_pat(cx, &param.pat, true, false);
    }

    fn check_arm(&mut self, cx: &EarlyContext<'_>, arm: &ast::Arm) {
        self.check_unused_parens_pat(cx, &arm.pat, false, false);
    }

    fn check_ty(&mut self, cx: &EarlyContext<'_>, ty: &ast::Ty) {
        if let &ast::TyKind::Paren(ref r) = &ty.kind {
            match &r.kind {
                &ast::TyKind::TraitObject(..) => {}
                &ast::TyKind::ImplTrait(_, ref bounds) if bounds.len() > 1 => {}
                &ast::TyKind::Array(_, ref len) => {
                    self.check_unused_delims_expr(
                        cx,
                        &len.value,
                        UnusedDelimsCtx::ArrayLenExpr,
                        false,
                        None,
                        None,
                    );
                }
                _ => {
                    let pattern_text =
                        if let Ok(snippet) = cx.sess().source_map().span_to_snippet(ty.span) {
                            snippet
                        } else {
                            pprust::ty_to_string(ty)
                        };

                    self.emit_unused_delims(cx, ty.span, &pattern_text, "type", (false, false));
                }
            }
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        <Self as UnusedDelimLint>::check_item(self, cx, item)
    }
}

declare_lint! {
    pub(super) UNUSED_BRACES,
    Warn,
    "unnecessary braces around an expression"
}

declare_lint_pass!(UnusedBraces => [UNUSED_BRACES]);

impl UnusedDelimLint for UnusedBraces {
    const DELIM_STR: &'static str = "braces";

    const LINT_EXPR_IN_PATTERN_MATCHING_CTX: bool = false;

    fn lint(&self) -> &'static Lint {
        UNUSED_BRACES
    }

    fn check_unused_delims_expr(
        &self,
        cx: &EarlyContext<'_>,
        value: &ast::Expr,
        ctx: UnusedDelimsCtx,
        followed_by_block: bool,
        left_pos: Option<BytePos>,
        right_pos: Option<BytePos>,
    ) {
        match value.kind {
            ast::ExprKind::Block(ref inner, None)
                if inner.rules == ast::BlockCheckMode::Default =>
            {
                // emit a warning under the following conditions:
                //
                // - the block does not have a label
                // - the block is not `unsafe`
                // - the block contains exactly one expression (do not lint `{ expr; }`)
                // - `followed_by_block` is true and the internal expr may contain a `{`
                // - the block is not multiline (do not lint multiline match arms)
                //      ```
                //      match expr {
                //          Pattern => {
                //              somewhat_long_expression
                //          }
                //          // ...
                //      }
                //      ```
                // - the block has no attribute and was not created inside a macro
                // - if the block is an `anon_const`, the inner expr must be a literal
                //      (do not lint `struct A<const N: usize>; let _: A<{ 2 + 3 }>;`)
                //
                // FIXME(const_generics): handle paths when #67075 is fixed.
                if let [stmt] = inner.stmts.as_slice() {
                    if let ast::StmtKind::Expr(ref expr) = stmt.kind {
                        if !Self::is_expr_delims_necessary(expr, followed_by_block)
                            && (ctx != UnusedDelimsCtx::AnonConst
                                || matches!(expr.kind, ast::ExprKind::Lit(_)))
                            // array length expressions are checked during `check_anon_const` and `check_ty`,
                            // once as `ArrayLenExpr` and once as `AnonConst`.
                            //
                            // As we do not want to lint this twice, we do not emit an error for
                            // `ArrayLenExpr` if `AnonConst` would do the same.
                            && (ctx != UnusedDelimsCtx::ArrayLenExpr
                                || !matches!(expr.kind, ast::ExprKind::Lit(_)))
                            && !cx.sess().source_map().is_multiline(value.span)
                            && value.attrs.is_empty()
                            && !value.span.from_expansion()
                        {
                            self.emit_unused_delims_expr(cx, value, ctx, left_pos, right_pos)
                        }
                    }
                }
            }
            ast::ExprKind::Let(_, ref expr) => {
                // FIXME(#60336): Properly handle `let true = (false && true)`
                // actually needing the parenthesis.
                self.check_unused_delims_expr(
                    cx,
                    expr,
                    UnusedDelimsCtx::LetScrutineeExpr,
                    followed_by_block,
                    None,
                    None,
                );
            }
            _ => {}
        }
    }
}

impl EarlyLintPass for UnusedBraces {
    fn check_expr(&mut self, cx: &EarlyContext<'_>, e: &ast::Expr) {
        <Self as UnusedDelimLint>::check_expr(self, cx, e)
    }

    fn check_anon_const(&mut self, cx: &EarlyContext<'_>, c: &ast::AnonConst) {
        self.check_unused_delims_expr(cx, &c.value, UnusedDelimsCtx::AnonConst, false, None, None);
    }

    fn check_stmt(&mut self, cx: &EarlyContext<'_>, s: &ast::Stmt) {
        <Self as UnusedDelimLint>::check_stmt(self, cx, s)
    }

    fn check_ty(&mut self, cx: &EarlyContext<'_>, ty: &ast::Ty) {
        if let &ast::TyKind::Paren(ref r) = &ty.kind {
            if let ast::TyKind::Array(_, ref len) = r.kind {
                self.check_unused_delims_expr(
                    cx,
                    &len.value,
                    UnusedDelimsCtx::ArrayLenExpr,
                    false,
                    None,
                    None,
                );
            }
        }
    }

    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        <Self as UnusedDelimLint>::check_item(self, cx, item)
    }
}

declare_lint! {
    UNUSED_IMPORT_BRACES,
    Allow,
    "unnecessary braces around an imported item"
}

declare_lint_pass!(UnusedImportBraces => [UNUSED_IMPORT_BRACES]);

impl UnusedImportBraces {
    fn check_use_tree(&self, cx: &EarlyContext<'_>, use_tree: &ast::UseTree, item: &ast::Item) {
        if let ast::UseTreeKind::Nested(ref items) = use_tree.kind {
            // Recursively check nested UseTrees
            for &(ref tree, _) in items {
                self.check_use_tree(cx, tree, item);
            }

            // Trigger the lint only if there is one nested item
            if items.len() != 1 {
                return;
            }

            // Trigger the lint if the nested item is a non-self single item
            let node_name = match items[0].0.kind {
                ast::UseTreeKind::Simple(rename, ..) => {
                    let orig_ident = items[0].0.prefix.segments.last().unwrap().ident;
                    if orig_ident.name == kw::SelfLower {
                        return;
                    }
                    rename.unwrap_or(orig_ident).name
                }
                ast::UseTreeKind::Glob => Symbol::intern("*"),
                ast::UseTreeKind::Nested(_) => return,
            };

            cx.struct_span_lint(UNUSED_IMPORT_BRACES, item.span, |lint| {
                lint.build(&format!("braces around {} is unnecessary", node_name)).emit()
            });
        }
    }
}

impl EarlyLintPass for UnusedImportBraces {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &ast::Item) {
        if let ast::ItemKind::Use(ref use_tree) = item.kind {
            self.check_use_tree(cx, use_tree, item);
        }
    }
}

declare_lint! {
    pub(super) UNUSED_ALLOCATION,
    Warn,
    "detects unnecessary allocations that can be eliminated"
}

declare_lint_pass!(UnusedAllocation => [UNUSED_ALLOCATION]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UnusedAllocation {
    fn check_expr(&mut self, cx: &LateContext<'_, '_>, e: &hir::Expr<'_>) {
        match e.kind {
            hir::ExprKind::Box(_) => {}
            _ => return,
        }

        for adj in cx.tables.expr_adjustments(e) {
            if let adjustment::Adjust::Borrow(adjustment::AutoBorrow::Ref(_, m)) = adj.kind {
                cx.struct_span_lint(UNUSED_ALLOCATION, e.span, |lint| {
                    let msg = match m {
                        adjustment::AutoBorrowMutability::Not => {
                            "unnecessary allocation, use `&` instead"
                        }
                        adjustment::AutoBorrowMutability::Mut { .. } => {
                            "unnecessary allocation, use `&mut` instead"
                        }
                    };
                    lint.build(msg).emit()
                });
            }
        }
    }
}
