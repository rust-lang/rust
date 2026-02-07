use rustc_ast::{BinOpKind, UnOp};
use rustc_hir::{self as hir, Expr, ExprKind, HirIdSet, LangItem, QPath, Stmt, StmtKind};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_middle::ty;
use rustc_session::lint::FutureIncompatibilityReason;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::{Span, sym};

use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `suspicious_cargo_cfg_target_family_comparisons` lint detects single-valued comparisons
    /// of [the `CARGO_CFG_TARGET_FAMILY`][CARGO_CFG_TARGET_FAMILY] environment variable.
    ///
    /// This variable is set by Cargo in build scripts.
    ///
    /// ### Example
    ///
    /// ```rust,no_run
    /// // build.rs
    /// fn main() {
    ///     let target_family = std::env::var("CARGO_CFG_TARGET_FAMILY").unwrap();
    ///
    ///     if target_family == "unix" {
    ///         // Do something specific to Unix platforms
    ///     }
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `CARGO_CFG_TARGET_FAMILY` is taken from [the `target_family` cfg][cfg-target_family], which
    /// may be set multiple times. This means that `CARGO_CFG_TARGET_FAMILY` can consist of multiple
    /// values, separated by commas. Comparing against a single value is thus not cross-platform.
    ///
    /// Note that most targets currently only have a single `target_family`, so oftentimes you
    /// wouldn't hit this. This is a [future-incompatible] lint, since the compiler may at some
    /// point introduce further target families for existing targets, and then a simple comparison
    /// would no longer work.
    ///
    /// [CARGO_CFG_TARGET_FAMILY]: https://doc.rust-lang.org/cargo/reference/environment-variables.html#:~:text=CARGO_CFG_TARGET_FAMILY
    /// [cfg-target_family]: https://doc.rust-lang.org/reference/conditional-compilation.html#target_family
    SUSPICIOUS_CARGO_CFG_TARGET_FAMILY_COMPARISONS,
    Warn,
    "comparing `CARGO_CFG_TARGET_FAMILY` env var with a single value",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::FutureReleaseSemanticsChange,
        reference: "issue #100343 <https://github.com/rust-lang/rust/issues/100343>",
        explain_reason: false,
    };
}

#[derive(Default)]
pub(crate) struct SuspiciousCargoCfgTargetFamilyComparisons {
    /// A side table of locals that are initialized from
    /// `std::env::var("CARGO_CFG_TARGET_FAMILY")` or similar.
    target_family_locals: HirIdSet,
}

impl_lint_pass!(SuspiciousCargoCfgTargetFamilyComparisons => [SUSPICIOUS_CARGO_CFG_TARGET_FAMILY_COMPARISONS]);

#[derive(LintDiagnostic)]
#[diag(lint_suspicious_cargo_cfg_target_family_comparison)]
#[note]
struct SingleValuedComparison {
    #[subdiagnostic]
    sugg: Option<ReplaceWithSplitAny>,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(lint_suggestion, applicability = "machine-applicable")]
struct ReplaceWithSplitAny {
    #[suggestion_part(code = "!")]
    negate: Option<Span>,
    #[suggestion_part(code = ".split(',').any(|x| x == ")]
    op: Span,
    #[suggestion_part(code = ")")]
    end: Span,
}

#[derive(LintDiagnostic)]
#[diag(lint_suspicious_cargo_cfg_target_family_match)]
#[note]
#[note(lint_suggestion)]
struct SingleValuedMatch;

// NOTE: We choose not to do a check for when in a build script, like:
// matches!(&sess.opts.crate_name, Some(crate_name) if crate_name == "build_script_build")
// Since we might be building a library that is used as a build script dependency (`cc-rs` etc).
impl<'tcx> LateLintPass<'tcx> for SuspiciousCargoCfgTargetFamilyComparisons {
    fn check_stmt(&mut self, _cx: &LateContext<'tcx>, stmt: &'tcx Stmt<'tcx>) {
        // Find locals that are initialized from `CARGO_CFG_TARGET_FAMILY`, and save them for later
        // checking.
        if let StmtKind::Let(stmt) = &stmt.kind {
            if let Some(init) = stmt.init {
                if self.accesses_target_family_env(init) {
                    stmt.pat.each_binding(|_, hir_id, _, _| {
                        self.target_family_locals.insert(hir_id);
                    });
                }
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) {
        // Check expressions that do single-valued comparisons.
        match &expr.kind {
            ExprKind::Binary(op, a, b) if matches!(op.node, BinOpKind::Eq | BinOpKind::Ne) => {
                if self.accesses_target_family_env(a) {
                    // If this is a &str or String, we can confidently give a `.split()` suggestion.
                    let a_ty = cx.typeck_results().expr_ty(a);
                    let is_str = matches!(
                        a_ty.kind(),
                        ty::Ref(_, r, _) if r.is_str(),
                    ) || matches!(
                        a_ty.ty_adt_def(),
                        Some(ty_def) if cx.tcx.is_lang_item(ty_def.did(), LangItem::String),
                    );
                    let sugg = is_str.then(|| ReplaceWithSplitAny {
                        negate: (op.node == BinOpKind::Ne).then(|| expr.span.shrink_to_lo()),
                        op: a.span.between(b.span),
                        end: b.span.shrink_to_hi(),
                    });

                    cx.emit_span_lint(
                        SUSPICIOUS_CARGO_CFG_TARGET_FAMILY_COMPARISONS,
                        expr.span,
                        SingleValuedComparison { sugg },
                    );
                } else if self.accesses_target_family_env(b) {
                    cx.emit_span_lint(
                        SUSPICIOUS_CARGO_CFG_TARGET_FAMILY_COMPARISONS,
                        expr.span,
                        // Unsure how to emit a suggestion when we need to reorder `a` and `b`.
                        SingleValuedComparison { sugg: None },
                    );
                }
            }
            ExprKind::Match(expr, _, _) if self.accesses_target_family_env(expr) => {
                cx.emit_span_lint(
                    SUSPICIOUS_CARGO_CFG_TARGET_FAMILY_COMPARISONS,
                    expr.span,
                    SingleValuedMatch,
                );
            }
            // We don't handle method calls like `PartialEq::eq`, that's probably fine though,
            // those are uncommon in real-world code.
            _ => {}
        }
    }
}

impl SuspiciousCargoCfgTargetFamilyComparisons {
    /// Check if an expression is likely derived from the `CARGO_CFG_TARGET_FAMILY` env var.
    fn accesses_target_family_env(&self, expr: &Expr<'_>) -> bool {
        match &expr.kind {
            // A call to `std::env::var[_os]("CARGO_CFG_TARGET_FAMILY")`.
            //
            // NOTE: This actually matches all functions that take as a single value
            // `"CARGO_CFG_TARGET_FAMILY"`. We could restrict this by matching only functions that
            // match `"std::env::var"` or `"std::env::var_os"` by doing something like:
            //
            // && let Expr { kind: ExprKind::Path(QPath::Resolved(_, path)), .. } = func
            // && let Some(fn_def_id) = path.res.opt_def_id()
            // && cx.tcx.is_diagnostic_item(sym::std_env_var, fn_def_id)
            //
            // But users often define wrapper functions around these, and so we wouldn't catch it
            // when they do.
            //
            // This is probably fine, `"CARGO_CFG_TARGET_FAMILY"` is unique enough of a name that
            // it's unlikely that people will be using it for anything else.
            ExprKind::Call(_, [arg])
                if let ExprKind::Lit(lit) = &arg.kind
                    && lit.node.str() == Some(sym::cargo_cfg_target_family) =>
            {
                true
            }
            // On local variables, try to see if it was initialized from target family earlier.
            ExprKind::Path(QPath::Resolved(_, path))
                if let hir::def::Res::Local(local_hir_id) = &path.res =>
            {
                self.target_family_locals.contains(local_hir_id)
            }
            // Recurse through references and dereferences.
            ExprKind::AddrOf(_, _, expr) | ExprKind::Unary(UnOp::Deref, expr) => {
                self.accesses_target_family_env(expr)
            }
            // Recurse on every method call to allow `.unwrap()`, `.as_deref()` and similar.
            //
            // NOTE: We could consider only recursing on specific `Option`/`Result` methods, but the
            // full list of the ones we'd want becomes unwieldy pretty quickly.
            ExprKind::MethodCall(_, receiver, _, _) => self.accesses_target_family_env(receiver),
            _ => false,
        }
    }
}
