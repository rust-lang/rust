use crate::internal_paths;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::paths::{PathNS, lookup_path};
use clippy_utils::{path_def_id, peel_ref_operators};
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_lint_defs::{declare_lint_pass, declare_tool_lint};
use rustc_middle::mir::ConstValue;
use rustc_span::symbol::Symbol;

declare_tool_lint! {
    /// ### What it does
    /// Checks for usage of def paths when a diagnostic item or a `LangItem` could be used.
    ///
    /// ### Why is this bad?
    /// The path for an item is subject to change and is less efficient to look up than a
    /// diagnostic item or a `LangItem`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// pub static VEC: PathLookup = path!(alloc::vec::Vec);
    ///
    /// VEC.contains_ty(cx, ty)
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// is_type_diagnostic_item(cx, ty, sym::Vec)
    /// ```
    pub clippy::UNNECESSARY_DEF_PATH,
    Warn,
    "using a def path when a diagnostic item or a `LangItem` is available",
    report_in_external_macro: true
}

declare_lint_pass!(UnnecessaryDefPath => [UNNECESSARY_DEF_PATH]);

impl<'tcx> LateLintPass<'tcx> for UnnecessaryDefPath {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Call(ctor, [_, path]) = expr.kind
            && internal_paths::PATH_LOOKUP_NEW.matches_path(cx, ctor)
            && let ExprKind::Array(segments) = peel_ref_operators(cx, path).kind
            && let Some(macro_id) = expr.span.ctxt().outer_expn_data().macro_def_id
        {
            let ns = match cx.tcx.item_name(macro_id).as_str() {
                "type_path" => PathNS::Type,
                "value_path" => PathNS::Value,
                "macro_path" => PathNS::Macro,
                _ => unreachable!(),
            };

            let path: Vec<Symbol> = segments
                .iter()
                .map(|segment| {
                    if let Some(const_def_id) = path_def_id(cx, segment)
                        && let Ok(ConstValue::Scalar(value)) = cx.tcx.const_eval_poly(const_def_id)
                        && let Some(value) = value.to_u32().discard_err()
                    {
                        Symbol::new(value)
                    } else {
                        panic!("failed to resolve path {:?}", expr.span);
                    }
                })
                .collect();

            for def_id in lookup_path(cx.tcx, ns, &path) {
                if let Some(name) = cx.tcx.get_diagnostic_name(def_id) {
                    span_lint_and_then(
                        cx,
                        UNNECESSARY_DEF_PATH,
                        expr.span.source_callsite(),
                        format!("a diagnostic name exists for this path: sym::{name}"),
                        |diag| {
                            diag.help(
                                "remove the `PathLookup` and use utilities such as `cx.tcx.is_diagnostic_item` instead",
                            );
                            diag.help("see also https://doc.rust-lang.org/nightly/nightly-rustc/?search=diag&filter-crate=clippy_utils");
                        },
                    );
                } else if let Some(item_name) = get_lang_item_name(cx, def_id) {
                    span_lint_and_then(
                        cx,
                        UNNECESSARY_DEF_PATH,
                        expr.span.source_callsite(),
                        format!("a language item exists for this path: LangItem::{item_name}"),
                        |diag| {
                            diag.help("remove the `PathLookup` and use utilities such as `cx.tcx.lang_items` instead");
                            diag.help("see also https://doc.rust-lang.org/nightly/nightly-rustc/?search=lang&filter-crate=clippy_utils");
                        },
                    );
                }
            }
        }
    }
}

fn get_lang_item_name(cx: &LateContext<'_>, def_id: DefId) -> Option<&'static str> {
    if let Some((lang_item, _)) = cx.tcx.lang_items().iter().find(|(_, id)| *id == def_id) {
        Some(lang_item.variant_name())
    } else {
        None
    }
}
