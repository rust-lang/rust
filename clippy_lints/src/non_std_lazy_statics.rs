use clippy_config::Conf;
use clippy_utils::diagnostics::{span_lint, span_lint_hir_and_then};
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::paths::{self, PathNS, find_crates, lookup_path_str};
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{fn_def_id, is_no_std_crate, path_def_id, sym};
use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::{self as hir, BodyId, Expr, ExprKind, HirId, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Lints when `once_cell::sync::Lazy` or `lazy_static!` are used to define a static variable,
    /// and suggests replacing such cases with `std::sync::LazyLock` instead.
    ///
    /// Note: This lint will not trigger in crate with `no_std` context, or with MSRV < 1.80.0. It
    /// also will not trigger on `once_cell::sync::Lazy` usage in crates which use other types
    /// from `once_cell`, such as `once_cell::race::OnceBox`.
    ///
    /// ### Why restrict this?
    /// - Reduces the need for an extra dependency
    /// - Enforce convention of using standard library types when possible
    ///
    /// ### Example
    /// ```ignore
    /// lazy_static! {
    ///     static ref FOO: String = "foo".to_uppercase();
    /// }
    /// static BAR: once_cell::sync::Lazy<String> = once_cell::sync::Lazy::new(|| "BAR".to_lowercase());
    /// ```
    /// Use instead:
    /// ```ignore
    /// static FOO: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| "FOO".to_lowercase());
    /// static BAR: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| "BAR".to_lowercase());
    /// ```
    #[clippy::version = "1.86.0"]
    pub NON_STD_LAZY_STATICS,
    pedantic,
    "lazy static that could be replaced by `std::sync::LazyLock`"
}

/// A list containing functions with corresponding replacements in `LazyLock`.
///
/// Some functions could be replaced as well if we have replaced `Lazy` to `LazyLock`,
/// therefore after suggesting replace the type, we need to make sure the function calls can be
/// replaced, otherwise the suggestions cannot be applied thus the applicability should be
/// `Unspecified` or `MaybeIncorret`.
static FUNCTION_REPLACEMENTS: &[(&str, Option<&str>)] = &[
    ("once_cell::sync::Lazy::force", Some("std::sync::LazyLock::force")),
    ("once_cell::sync::Lazy::get", None), // `std::sync::LazyLock::get` is experimental
    ("once_cell::sync::Lazy::new", Some("std::sync::LazyLock::new")),
    // Note: `Lazy::{into_value, get_mut, force_mut}` are not in the list.
    // Because the lint only checks for `static`s, and using these functions with statics
    // will either be a hard error or triggers `static_mut_ref` that will be hard errors.
    // But keep in mind that if somehow we decide to expand this lint to catch non-statics,
    // add those functions into the list.
];

pub struct NonStdLazyStatic {
    msrv: Msrv,
    once_cell_crates: Vec<CrateNum>,
    sugg_map: FxIndexMap<DefId, Option<String>>,
    lazy_type_defs: FxIndexMap<DefId, LazyInfo>,
    uses_other_once_cell_types: bool,
}

impl NonStdLazyStatic {
    #[must_use]
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            once_cell_crates: Vec::new(),
            sugg_map: FxIndexMap::default(),
            lazy_type_defs: FxIndexMap::default(),
            uses_other_once_cell_types: false,
        }
    }
}

impl_lint_pass!(NonStdLazyStatic => [NON_STD_LAZY_STATICS]);

fn can_use_lazy_cell(cx: &LateContext<'_>, msrv: Msrv) -> bool {
    msrv.meets(cx, msrvs::LAZY_CELL) && !is_no_std_crate(cx)
}

impl<'hir> LateLintPass<'hir> for NonStdLazyStatic {
    fn check_crate(&mut self, cx: &LateContext<'hir>) {
        // Add CrateNums for `once_cell` crate
        self.once_cell_crates = find_crates(cx.tcx, sym::once_cell)
            .iter()
            .map(|def_id| def_id.krate)
            .collect();

        // Convert hardcoded fn replacement list into a map with def_id
        for (path, sugg) in FUNCTION_REPLACEMENTS {
            for did in lookup_path_str(cx.tcx, PathNS::Value, path) {
                self.sugg_map.insert(did, sugg.map(ToOwned::to_owned));
            }
        }
    }

    fn check_item(&mut self, cx: &LateContext<'hir>, item: &Item<'hir>) {
        if let ItemKind::Static(..) = item.kind
            && let Some(macro_call) = clippy_utils::macros::root_macro_call(item.span)
            && paths::LAZY_STATIC.matches(cx, macro_call.def_id)
            && can_use_lazy_cell(cx, self.msrv)
        {
            span_lint(
                cx,
                NON_STD_LAZY_STATICS,
                macro_call.span,
                "this macro has been superseded by `std::sync::LazyLock`",
            );
            return;
        }

        if item.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        if let Some(lazy_info) = LazyInfo::from_item(cx, item)
            && can_use_lazy_cell(cx, self.msrv)
        {
            self.lazy_type_defs.insert(item.owner_id.to_def_id(), lazy_info);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'hir>, expr: &Expr<'hir>) {
        // All functions in the `FUNCTION_REPLACEMENTS` have only one args
        if let ExprKind::Call(callee, [arg]) = expr.kind
            && let Some(call_def_id) = fn_def_id(cx, expr)
            && self.sugg_map.contains_key(&call_def_id)
            && let ExprKind::Path(qpath) = arg.peel_borrows().kind
            && let Some(arg_def_id) = cx.typeck_results().qpath_res(&qpath, arg.hir_id).opt_def_id()
            && let Some(lazy_info) = self.lazy_type_defs.get_mut(&arg_def_id)
        {
            lazy_info.calls_span_and_id.insert(callee.span, call_def_id);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'hir>, ty: &'hir rustc_hir::Ty<'hir, rustc_hir::AmbigArg>) {
        // Record if types from `once_cell` besides `sync::Lazy` are used.
        if let rustc_hir::TyKind::Path(qpath) = ty.peel_refs().kind
            && let Some(ty_def_id) = cx.qpath_res(&qpath, ty.hir_id).opt_def_id()
            // Is from `once_cell` crate
            && self.once_cell_crates.contains(&ty_def_id.krate)
            // And is NOT `once_cell::sync::Lazy`
            && !paths::ONCE_CELL_SYNC_LAZY.matches(cx, ty_def_id)
        {
            self.uses_other_once_cell_types = true;
        }
    }

    fn check_crate_post(&mut self, cx: &LateContext<'hir>) {
        if !self.uses_other_once_cell_types {
            for (_, lazy_info) in &self.lazy_type_defs {
                lazy_info.lint(cx, &self.sugg_map);
            }
        }
    }
}

struct LazyInfo {
    /// Span of the [`hir::Ty`] without including args.
    /// i.e.:
    /// ```ignore
    /// static FOO: Lazy<String> = Lazy::new(...);
    /// //          ^^^^
    /// ```
    ty_span_no_args: Span,
    /// Item on which the lint must be generated.
    item_hir_id: HirId,
    /// `Span` and `DefId` of calls on `Lazy` type.
    /// i.e.:
    /// ```ignore
    /// static FOO: Lazy<String> = Lazy::new(...);
    /// //                         ^^^^^^^^^
    /// ```
    calls_span_and_id: FxIndexMap<Span, DefId>,
}

impl LazyInfo {
    fn from_item(cx: &LateContext<'_>, item: &Item<'_>) -> Option<Self> {
        // Check if item is a `once_cell:sync::Lazy` static.
        if let ItemKind::Static(_, ty, _, body_id) = item.kind
            && let Some(path_def_id) = path_def_id(cx, ty)
            && let hir::TyKind::Path(hir::QPath::Resolved(_, path)) = ty.kind
            && paths::ONCE_CELL_SYNC_LAZY.matches(cx, path_def_id)
        {
            let ty_span_no_args = path_span_without_args(path);
            let body = cx.tcx.hir_body(body_id);

            // visit body to collect `Lazy::new` calls
            let mut new_fn_calls = FxIndexMap::default();
            for_each_expr::<(), ()>(cx, body, |ex| {
                if let Some((fn_did, call_span)) = fn_def_id_and_span_from_body(cx, ex, body_id)
                    && paths::ONCE_CELL_SYNC_LAZY_NEW.matches(cx, fn_did)
                {
                    new_fn_calls.insert(call_span, fn_did);
                }
                std::ops::ControlFlow::Continue(())
            });

            Some(LazyInfo {
                ty_span_no_args,
                item_hir_id: item.hir_id(),
                calls_span_and_id: new_fn_calls,
            })
        } else {
            None
        }
    }

    fn lint(&self, cx: &LateContext<'_>, sugg_map: &FxIndexMap<DefId, Option<String>>) {
        // Applicability might get adjusted to `Unspecified` later if any calls
        // in `calls_span_and_id` are not replaceable judging by the `sugg_map`.
        let mut appl = Applicability::MachineApplicable;
        let mut suggs = vec![(self.ty_span_no_args, "std::sync::LazyLock".to_string())];

        for (span, def_id) in &self.calls_span_and_id {
            let maybe_sugg = sugg_map.get(def_id).cloned().flatten();
            if let Some(sugg) = maybe_sugg {
                suggs.push((*span, sugg));
            } else {
                // If NO suggested replacement, not machine applicable
                appl = Applicability::Unspecified;
            }
        }

        span_lint_hir_and_then(
            cx,
            NON_STD_LAZY_STATICS,
            self.item_hir_id,
            self.ty_span_no_args,
            "this type has been superseded by `LazyLock` in the standard library",
            |diag| {
                diag.multipart_suggestion("use `std::sync::LazyLock` instead", suggs, appl);
            },
        );
    }
}

/// Return the span of a given `Path` without including any of its args.
///
/// NB: Re-write of a private function `rustc_lint::non_local_def::path_span_without_args`.
fn path_span_without_args(path: &hir::Path<'_>) -> Span {
    path.segments
        .last()
        .and_then(|seg| seg.args)
        .map_or(path.span, |args| path.span.until(args.span_ext))
}

/// Returns the `DefId` and `Span` of the callee if the given expression is a function call.
///
/// NB: Modified from [`clippy_utils::fn_def_id`], to support calling in an static `Item`'s body.
fn fn_def_id_and_span_from_body(cx: &LateContext<'_>, expr: &Expr<'_>, body_id: BodyId) -> Option<(DefId, Span)> {
    // FIXME: find a way to cache the result.
    let typeck = cx.tcx.typeck_body(body_id);
    match &expr.kind {
        ExprKind::Call(
            Expr {
                kind: ExprKind::Path(qpath),
                hir_id: path_hir_id,
                span,
                ..
            },
            ..,
        ) => {
            // Only return Fn-like DefIds, not the DefIds of statics/consts/etc that contain or
            // deref to fn pointers, dyn Fn, impl Fn - #8850
            if let Res::Def(DefKind::Fn | DefKind::Ctor(..) | DefKind::AssocFn, id) =
                typeck.qpath_res(qpath, *path_hir_id)
            {
                Some((id, *span))
            } else {
                None
            }
        },
        _ => None,
    }
}
