use regex_syntax;
use rustc::hir::*;
use rustc::lint::*;
use rustc::middle::const_val::ConstVal;
use rustc_const_eval::EvalHint::ExprTypeChecked;
use rustc_const_eval::eval_const_expr_partial;
use std::collections::HashSet;
use std::error::Error;
use syntax::ast::{LitKind, NodeId};
use syntax::codemap::{Span, BytePos};
use syntax::parse::token::InternedString;
use utils::{is_expn_of, match_def_path, match_type, paths, span_lint, span_help_and_lint};

/// **What it does:** This lint checks [regex] creation (with `Regex::new`, `RegexBuilder::new` or
/// `RegexSet::new`) for correct regex syntax.
///
/// **Why is this bad?** This will lead to a runtime panic.
///
/// **Known problems:** None.
///
/// **Example:** `Regex::new("|")`
declare_lint! {
    pub INVALID_REGEX,
    Deny,
    "finds invalid regular expressions"
}

/// **What it does:** This lint checks for trivial [regex] creation (with `Regex::new`,
/// `RegexBuilder::new` or `RegexSet::new`).
///
/// **Why is this bad?** This can likely be replaced by `==` or `str::starts_with`,
/// `str::ends_with` or `std::contains` or other `str` methods.
///
/// **Known problems:** None.
///
/// **Example:** `Regex::new("^foobar")`
///
/// [regex]: https://crates.io/crates/regex
declare_lint! {
    pub TRIVIAL_REGEX,
    Warn,
    "finds trivial regular expressions"
}

/// **What it does:** This lint checks for usage of `regex!(_)` which as of now is usually slower than `Regex::new(_)` unless called in a loop (which is a bad idea anyway).
///
/// **Why is this bad?** Performance, at least for now. The macro version is likely to catch up long-term, but for now the dynamic version is faster.
///
/// **Known problems:** None
///
/// **Example:** `regex!("foo|bar")`
declare_lint! {
    pub REGEX_MACRO,
    Warn,
    "finds use of `regex!(_)`, suggests `Regex::new(_)` instead"
}

#[derive(Clone, Default)]
pub struct RegexPass {
    spans: HashSet<Span>,
    last: Option<NodeId>,
}

impl LintPass for RegexPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(INVALID_REGEX, REGEX_MACRO, TRIVIAL_REGEX)
    }
}

impl LateLintPass for RegexPass {
    fn check_crate(&mut self, _: &LateContext, _: &Crate) {
        self.spans.clear();
    }

    fn check_block(&mut self, cx: &LateContext, block: &Block) {
        if_let_chain!{[
            self.last.is_none(),
            let Some(ref expr) = block.expr,
            match_type(cx, cx.tcx.expr_ty(expr), &paths::REGEX),
            let Some(span) = is_expn_of(cx, expr.span, "regex"),
        ], {
            if !self.spans.contains(&span) {
                span_lint(cx,
                          REGEX_MACRO,
                          span,
                          "`regex!(_)` found. \
                          Please use `Regex::new(_)`, which is faster for now.");
                self.spans.insert(span);
            }
            self.last = Some(block.id);
        }}
    }

    fn check_block_post(&mut self, _: &LateContext, block: &Block) {
        if self.last.map_or(false, |id| block.id == id) {
            self.last = None;
        }
    }

    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain!{[
            let ExprCall(ref fun, ref args) = expr.node,
            args.len() == 1,
            let Some(def) = cx.tcx.def_map.borrow().get(&fun.id),
        ], {
            let def_id = def.def_id();
            if match_def_path(cx, def_id, &paths::REGEX_NEW) ||
               match_def_path(cx, def_id, &paths::REGEX_BUILDER_NEW) {
                check_regex(cx, &args[0], true);
            } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_NEW) ||
               match_def_path(cx, def_id, &paths::REGEX_BYTES_BUILDER_NEW) {
                check_regex(cx, &args[0], false);
            } else if match_def_path(cx, def_id, &paths::REGEX_SET_NEW) {
                check_set(cx, &args[0], true);
            } else if match_def_path(cx, def_id, &paths::REGEX_BYTES_SET_NEW) {
                check_set(cx, &args[0], false);
            }
        }}
    }
}

#[allow(cast_possible_truncation)]
fn str_span(base: Span, s: &str, c: usize) -> Span {
    let mut si = s.char_indices().skip(c);

    match (si.next(), si.next()) {
        (Some((l, _)), Some((h, _))) => {
            Span {
                lo: base.lo + BytePos(l as u32),
                hi: base.lo + BytePos(h as u32),
                ..base
            }
        }
        _ => base,
    }
}

fn const_str(cx: &LateContext, e: &Expr) -> Option<InternedString> {
    match eval_const_expr_partial(cx.tcx, e, ExprTypeChecked, None) {
        Ok(ConstVal::Str(r)) => Some(r),
        _ => None,
    }
}

fn is_trivial_regex(s: &regex_syntax::Expr) -> Option<&'static str> {
    use regex_syntax::Expr;

    match *s {
        Expr::Empty | Expr::StartText | Expr::EndText => Some("the regex is unlikely to be useful as it is"),
        Expr::Literal { .. } => Some("consider using `str::contains`"),
        Expr::Concat(ref exprs) => {
            match exprs.len() {
                2 => {
                    match (&exprs[0], &exprs[1]) {
                        (&Expr::StartText, &Expr::EndText) => Some("consider using `str::is_empty`"),
                        (&Expr::StartText, &Expr::Literal { .. }) => Some("consider using `str::starts_with`"),
                        (&Expr::Literal { .. }, &Expr::EndText) => Some("consider using `str::ends_with`"),
                        _ => None,
                    }
                }
                3 => {
                    if let (&Expr::StartText, &Expr::Literal {..}, &Expr::EndText) = (&exprs[0], &exprs[1], &exprs[2]) {
                        Some("consider using `==` on `str`s")
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn check_set(cx: &LateContext, expr: &Expr, utf8: bool) {
    if_let_chain! {[
        let ExprAddrOf(_, ref expr) = expr.node,
        let ExprVec(ref exprs) = expr.node,
    ], {
        for expr in exprs {
            check_regex(cx, expr, utf8);
        }
    }}
}

fn check_regex(cx: &LateContext, expr: &Expr, utf8: bool) {
    let builder = regex_syntax::ExprBuilder::new().unicode(utf8);

    if let ExprLit(ref lit) = expr.node {
        if let LitKind::Str(ref r, _) = lit.node {
            match builder.parse(r) {
                Ok(r) => {
                    if let Some(repl) = is_trivial_regex(&r) {
                        span_help_and_lint(cx,
                                           TRIVIAL_REGEX,
                                           expr.span,
                                           "trivial regex",
                                           &format!("consider using {}", repl));
                    }
                }
                Err(e) => {
                    span_lint(cx,
                              INVALID_REGEX,
                              str_span(expr.span, r, e.position()),
                              &format!("regex syntax error: {}",
                                       e.description()));
                }
            }
        }
    } else if let Some(r) = const_str(cx, expr) {
        match builder.parse(&r) {
            Ok(r) => {
                if let Some(repl) = is_trivial_regex(&r) {
                    span_help_and_lint(cx,
                                       TRIVIAL_REGEX,
                                       expr.span,
                                       "trivial regex",
                                       &format!("consider using {}", repl));
                }
            }
            Err(e) => {
                span_lint(cx,
                          INVALID_REGEX,
                          expr.span,
                          &format!("regex syntax error on position {}: {}",
                                   e.position(),
                                   e.description()));
            }
        }
    }
}
