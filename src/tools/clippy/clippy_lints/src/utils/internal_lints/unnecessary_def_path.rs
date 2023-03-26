use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{def_path_def_ids, is_lint_allowed, match_any_def_paths, peel_hir_expr_refs};
use if_chain::if_chain;
use rustc_ast::ast::LitKind;
use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::{Expr, ExprKind, Local, Mutability, Node};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::mir::interpret::{Allocation, ConstValue, GlobalAlloc};
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::Symbol;
use rustc_span::Span;

use std::str;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usages of def paths when a diagnostic item or a `LangItem` could be used.
    ///
    /// ### Why is this bad?
    /// The path for an item is subject to change and is less efficient to look up than a
    /// diagnostic item or a `LangItem`.
    ///
    /// ### Example
    /// ```rust,ignore
    /// utils::match_type(cx, ty, &paths::VEC)
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// utils::is_type_diagnostic_item(cx, ty, sym::Vec)
    /// ```
    pub UNNECESSARY_DEF_PATH,
    internal,
    "using a def path when a diagnostic item or a `LangItem` is available"
}

impl_lint_pass!(UnnecessaryDefPath => [UNNECESSARY_DEF_PATH]);

#[derive(Default)]
pub struct UnnecessaryDefPath {
    array_def_ids: FxIndexSet<(DefId, Span)>,
    linted_def_ids: FxHashSet<DefId>,
}

impl<'tcx> LateLintPass<'tcx> for UnnecessaryDefPath {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx hir::Expr<'_>) {
        if is_lint_allowed(cx, UNNECESSARY_DEF_PATH, expr.hir_id) {
            return;
        }

        match expr.kind {
            ExprKind::Call(func, args) => self.check_call(cx, func, args, expr.span),
            ExprKind::Array(elements) => self.check_array(cx, elements, expr.span),
            _ => {},
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        for &(def_id, span) in &self.array_def_ids {
            if self.linted_def_ids.contains(&def_id) {
                continue;
            }

            let (msg, sugg) = if let Some(sym) = cx.tcx.get_diagnostic_name(def_id) {
                ("diagnostic item", format!("sym::{sym}"))
            } else if let Some(sym) = get_lang_item_name(cx, def_id) {
                ("language item", format!("LangItem::{sym}"))
            } else {
                continue;
            };

            span_lint_and_help(
                cx,
                UNNECESSARY_DEF_PATH,
                span,
                &format!("hardcoded path to a {msg}"),
                None,
                &format!("convert all references to use `{sugg}`"),
            );
        }
    }
}

impl UnnecessaryDefPath {
    #[allow(clippy::too_many_lines)]
    fn check_call(&mut self, cx: &LateContext<'_>, func: &Expr<'_>, args: &[Expr<'_>], span: Span) {
        enum Item {
            LangItem(&'static str),
            DiagnosticItem(Symbol),
        }
        static PATHS: &[&[&str]] = &[
            &["clippy_utils", "match_def_path"],
            &["clippy_utils", "match_trait_method"],
            &["clippy_utils", "ty", "match_type"],
            &["clippy_utils", "is_expr_path_def_path"],
        ];

        if_chain! {
            if let [cx_arg, def_arg, args @ ..] = args;
            if let ExprKind::Path(path) = &func.kind;
            if let Some(id) = cx.qpath_res(path, func.hir_id).opt_def_id();
            if let Some(which_path) = match_any_def_paths(cx, id, PATHS);
            let item_arg = if which_path == 4 { &args[1] } else { &args[0] };
            // Extract the path to the matched type
            if let Some(segments) = path_to_matched_type(cx, item_arg);
            let segments: Vec<&str> = segments.iter().map(|sym| &**sym).collect();
            if let Some(def_id) = def_path_def_ids(cx, &segments[..]).next();
            then {
                // Check if the target item is a diagnostic item or LangItem.
                #[rustfmt::skip]
                let (msg, item) = if let Some(item_name)
                    = cx.tcx.diagnostic_items(def_id.krate).id_to_name.get(&def_id)
                {
                    (
                        "use of a def path to a diagnostic item",
                        Item::DiagnosticItem(*item_name),
                    )
                } else if let Some(item_name) = get_lang_item_name(cx, def_id) {
                    (
                        "use of a def path to a `LangItem`",
                        Item::LangItem(item_name),
                    )
                } else {
                    return;
                };

                let has_ctor = match cx.tcx.def_kind(def_id) {
                    DefKind::Struct => {
                        let variant = cx.tcx.adt_def(def_id).non_enum_variant();
                        variant.ctor.is_some() && variant.fields.iter().all(|f| f.vis.is_public())
                    },
                    DefKind::Variant => {
                        let variant = cx.tcx.adt_def(cx.tcx.parent(def_id)).variant_with_id(def_id);
                        variant.ctor.is_some() && variant.fields.iter().all(|f| f.vis.is_public())
                    },
                    _ => false,
                };

                let mut app = Applicability::MachineApplicable;
                let cx_snip = snippet_with_applicability(cx, cx_arg.span, "..", &mut app);
                let def_snip = snippet_with_applicability(cx, def_arg.span, "..", &mut app);
                let (sugg, with_note) = match (which_path, item) {
                    // match_def_path
                    (0, Item::DiagnosticItem(item)) => (
                        format!("{cx_snip}.tcx.is_diagnostic_item(sym::{item}, {def_snip})"),
                        has_ctor,
                    ),
                    (0, Item::LangItem(item)) => (
                        format!("{cx_snip}.tcx.lang_items().get(LangItem::{item}) == Some({def_snip})"),
                        has_ctor,
                    ),
                    // match_trait_method
                    (1, Item::DiagnosticItem(item)) => {
                        (format!("is_trait_method({cx_snip}, {def_snip}, sym::{item})"), false)
                    },
                    // match_type
                    (2, Item::DiagnosticItem(item)) => (
                        format!("is_type_diagnostic_item({cx_snip}, {def_snip}, sym::{item})"),
                        false,
                    ),
                    (2, Item::LangItem(item)) => (
                        format!("is_type_lang_item({cx_snip}, {def_snip}, LangItem::{item})"),
                        false,
                    ),
                    // is_expr_path_def_path
                    (3, Item::DiagnosticItem(item)) if has_ctor => (
                        format!("is_res_diag_ctor({cx_snip}, path_res({cx_snip}, {def_snip}), sym::{item})",),
                        false,
                    ),
                    (3, Item::LangItem(item)) if has_ctor => (
                        format!("is_res_lang_ctor({cx_snip}, path_res({cx_snip}, {def_snip}), LangItem::{item})",),
                        false,
                    ),
                    (3, Item::DiagnosticItem(item)) => (
                        format!("is_path_diagnostic_item({cx_snip}, {def_snip}, sym::{item})"),
                        false,
                    ),
                    (3, Item::LangItem(item)) => (
                        format!(
                            "path_res({cx_snip}, {def_snip}).opt_def_id()\
                                .map_or(false, |id| {cx_snip}.tcx.lang_items().get(LangItem::{item}) == Some(id))",
                        ),
                        false,
                    ),
                    _ => return,
                };

                span_lint_and_then(cx, UNNECESSARY_DEF_PATH, span, msg, |diag| {
                    diag.span_suggestion(span, "try", sugg, app);
                    if with_note {
                        diag.help(
                            "if this `DefId` came from a constructor expression or pattern then the \
                                    parent `DefId` should be used instead",
                        );
                    }
                });

                self.linted_def_ids.insert(def_id);
            }
        }
    }

    fn check_array(&mut self, cx: &LateContext<'_>, elements: &[Expr<'_>], span: Span) {
        let Some(path) = path_from_array(elements) else { return };

        for def_id in def_path_def_ids(cx, &path.iter().map(AsRef::as_ref).collect::<Vec<_>>()) {
            self.array_def_ids.insert((def_id, span));
        }
    }
}

fn path_to_matched_type(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> Option<Vec<String>> {
    match peel_hir_expr_refs(expr).0.kind {
        ExprKind::Path(ref qpath) => match cx.qpath_res(qpath, expr.hir_id) {
            Res::Local(hir_id) => {
                let parent_id = cx.tcx.hir().parent_id(hir_id);
                if let Some(Node::Local(Local { init: Some(init), .. })) = cx.tcx.hir().find(parent_id) {
                    path_to_matched_type(cx, init)
                } else {
                    None
                }
            },
            Res::Def(DefKind::Static(_), def_id) => read_mir_alloc_def_path(
                cx,
                cx.tcx.eval_static_initializer(def_id).ok()?.inner(),
                cx.tcx.type_of(def_id).subst_identity(),
            ),
            Res::Def(DefKind::Const, def_id) => match cx.tcx.const_eval_poly(def_id).ok()? {
                ConstValue::ByRef { alloc, offset } if offset.bytes() == 0 => {
                    read_mir_alloc_def_path(cx, alloc.inner(), cx.tcx.type_of(def_id).subst_identity())
                },
                _ => None,
            },
            _ => None,
        },
        ExprKind::Array(exprs) => path_from_array(exprs),
        _ => None,
    }
}

fn read_mir_alloc_def_path<'tcx>(cx: &LateContext<'tcx>, alloc: &'tcx Allocation, ty: Ty<'_>) -> Option<Vec<String>> {
    let (alloc, ty) = if let ty::Ref(_, ty, Mutability::Not) = *ty.kind() {
        let &alloc = alloc.provenance().ptrs().values().next()?;
        if let GlobalAlloc::Memory(alloc) = cx.tcx.global_alloc(alloc) {
            (alloc.inner(), ty)
        } else {
            return None;
        }
    } else {
        (alloc, ty)
    };

    if let ty::Array(ty, _) | ty::Slice(ty) = *ty.kind()
        && let ty::Ref(_, ty, Mutability::Not) = *ty.kind()
        && ty.is_str()
    {
        alloc
            .provenance()
            .ptrs()
            .values()
            .map(|&alloc| {
                if let GlobalAlloc::Memory(alloc) = cx.tcx.global_alloc(alloc) {
                    let alloc = alloc.inner();
                    str::from_utf8(alloc.inspect_with_uninit_and_ptr_outside_interpreter(0..alloc.len()))
                        .ok().map(ToOwned::to_owned)
                } else {
                    None
                }
            })
            .collect()
    } else {
        None
    }
}

fn path_from_array(exprs: &[Expr<'_>]) -> Option<Vec<String>> {
    exprs
        .iter()
        .map(|expr| {
            if let ExprKind::Lit(lit) = &expr.kind {
                if let LitKind::Str(sym, _) = lit.node {
                    return Some((*sym.as_str()).to_owned());
                }
            }

            None
        })
        .collect()
}

fn get_lang_item_name(cx: &LateContext<'_>, def_id: DefId) -> Option<&'static str> {
    if let Some((lang_item, _)) = cx.tcx.lang_items().iter().find(|(_, id)| *id == def_id) {
        Some(lang_item.variant_name())
    } else {
        None
    }
}
