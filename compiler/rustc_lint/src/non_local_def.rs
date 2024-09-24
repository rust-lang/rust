use rustc_errors::MultiSpan;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{Body, HirId, Item, ItemKind, Node, Path, TyKind};
use rustc_middle::ty::TyCtxt;
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::symbol::kw;
use rustc_span::{ExpnKind, MacroKind, Span, sym};

use crate::lints::{NonLocalDefinitionsCargoUpdateNote, NonLocalDefinitionsDiag};
use crate::{LateContext, LateLintPass, LintContext, fluent_generated as fluent};

declare_lint! {
    /// The `non_local_definitions` lint checks for `impl` blocks and `#[macro_export]`
    /// macro inside bodies (functions, enum discriminant, ...).
    ///
    /// ### Example
    ///
    /// ```rust
    /// #![warn(non_local_definitions)]
    /// trait MyTrait {}
    /// struct MyStruct;
    ///
    /// fn foo() {
    ///     impl MyTrait for MyStruct {}
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Creating non-local definitions go against expectation and can create discrepancies
    /// in tooling. It should be avoided. It may become deny-by-default in edition 2024
    /// and higher, see the tracking issue <https://github.com/rust-lang/rust/issues/120363>.
    ///
    /// An `impl` definition is non-local if it is nested inside an item and neither
    /// the type nor the trait are at the same nesting level as the `impl` block.
    ///
    /// All nested bodies (functions, enum discriminant, array length, consts) (expect for
    /// `const _: Ty = { ... }` in top-level module, which is still undecided) are checked.
    pub NON_LOCAL_DEFINITIONS,
    Warn,
    "checks for non-local definitions",
    report_in_external_macro
}

#[derive(Default)]
pub(crate) struct NonLocalDefinitions {
    body_depth: u32,
}

impl_lint_pass!(NonLocalDefinitions => [NON_LOCAL_DEFINITIONS]);

// FIXME(Urgau): Figure out how to handle modules nested in bodies.
// It's currently not handled by the current logic because modules are not bodies.
// They don't even follow the correct order (check_body -> check_mod -> check_body_post)
// instead check_mod is called after every body has been handled.

impl<'tcx> LateLintPass<'tcx> for NonLocalDefinitions {
    fn check_body(&mut self, _cx: &LateContext<'tcx>, _body: &Body<'tcx>) {
        self.body_depth += 1;
    }

    fn check_body_post(&mut self, _cx: &LateContext<'tcx>, _body: &Body<'tcx>) {
        self.body_depth -= 1;
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if self.body_depth == 0 {
            return;
        }

        let def_id = item.owner_id.def_id.into();
        let parent = cx.tcx.parent(def_id);
        let parent_def_kind = cx.tcx.def_kind(parent);
        let parent_opt_item_name = cx.tcx.opt_item_name(parent);

        // Per RFC we (currently) ignore anon-const (`const _: Ty = ...`) in top-level module.
        if self.body_depth == 1
            && parent_def_kind == DefKind::Const
            && parent_opt_item_name == Some(kw::Underscore)
        {
            return;
        }

        let cargo_update = || {
            let oexpn = item.span.ctxt().outer_expn_data();
            if let Some(def_id) = oexpn.macro_def_id
                && let ExpnKind::Macro(macro_kind, macro_name) = oexpn.kind
                && def_id.krate != LOCAL_CRATE
                && rustc_session::utils::was_invoked_from_cargo()
            {
                Some(NonLocalDefinitionsCargoUpdateNote {
                    macro_kind: macro_kind.descr(),
                    macro_name,
                    crate_name: cx.tcx.crate_name(def_id.krate),
                })
            } else {
                None
            }
        };

        // determining if we are in a doctest context can't currently be determined
        // by the code itself (there are no specific attributes), but fortunately rustdoc
        // sets a perma-unstable env var for libtest so we just reuse that for now
        let is_at_toplevel_doctest =
            || self.body_depth == 2 && std::env::var("UNSTABLE_RUSTDOC_TEST_PATH").is_ok();

        match item.kind {
            ItemKind::Impl(impl_) => {
                // The RFC states:
                //
                // > An item nested inside an expression-containing item (through any
                // > level of nesting) may not define an impl Trait for Type unless
                // > either the **Trait** or the **Type** is also nested inside the
                // > same expression-containing item.
                //
                // To achieve this we get try to get the paths of the _Trait_ and
                // _Type_, and we look inside those paths to try a find in one
                // of them a type whose parent is the same as the impl definition.
                //
                // If that's the case this means that this impl block declaration
                // is using local items and so we don't lint on it.

                // We also ignore anon-const in item by including the anon-const
                // parent as well.
                let parent_parent = if parent_def_kind == DefKind::Const
                    && parent_opt_item_name == Some(kw::Underscore)
                {
                    Some(cx.tcx.parent(parent))
                } else {
                    None
                };

                // 1. We collect all the `hir::Path` from the `Self` type and `Trait` ref
                // of the `impl` definition
                let mut collector = PathCollector { paths: Vec::new() };
                collector.visit_ty(&impl_.self_ty);
                if let Some(of_trait) = &impl_.of_trait {
                    collector.visit_trait_ref(of_trait);
                }

                // 1.5. Remove any path that doesn't resolve to a `DefId` or if it resolve to a
                // type-param (e.g. `T`).
                collector.paths.retain(
                    |p| matches!(p.res, Res::Def(def_kind, _) if def_kind != DefKind::TyParam),
                );

                // 2. We check if any of path reference a "local" parent and if that the case
                // we bail out as asked by T-lang, even though this isn't correct from a
                // type-system point of view, as inference exists and could still leak the impl.
                if collector
                    .paths
                    .iter()
                    .any(|path| path_has_local_parent(path, cx, parent, parent_parent))
                {
                    return;
                }

                // Get the span of the parent const item ident (if it's a not a const anon).
                //
                // Used to suggest changing the const item to a const anon.
                let span_for_const_anon_suggestion = if parent_def_kind == DefKind::Const
                    && parent_opt_item_name != Some(kw::Underscore)
                    && let Some(parent) = parent.as_local()
                    && let Node::Item(item) = cx.tcx.hir_node_by_def_id(parent)
                    && let ItemKind::Const(ty, _, _) = item.kind
                    && let TyKind::Tup(&[]) = ty.kind
                {
                    Some(item.ident.span)
                } else {
                    None
                };

                let const_anon = matches!(parent_def_kind, DefKind::Const | DefKind::Static { .. })
                    .then_some(span_for_const_anon_suggestion);

                let impl_span = item.span.shrink_to_lo().to(impl_.self_ty.span);
                let mut ms = MultiSpan::from_span(impl_span);

                for path in &collector.paths {
                    // FIXME: While a translatable diagnostic message can have an argument
                    // we (currently) have no way to set different args per diag msg with
                    // `MultiSpan::push_span_label`.
                    #[allow(rustc::untranslatable_diagnostic)]
                    ms.push_span_label(
                        path_span_without_args(path),
                        format!("`{}` is not local", path_name_to_string(path)),
                    );
                }

                let doctest = is_at_toplevel_doctest();

                if !doctest {
                    ms.push_span_label(
                        cx.tcx.def_span(parent),
                        fluent::lint_non_local_definitions_impl_move_help,
                    );
                }

                let macro_to_change =
                    if let ExpnKind::Macro(kind, name) = item.span.ctxt().outer_expn_data().kind {
                        Some((name.to_string(), kind.descr()))
                    } else {
                        None
                    };

                cx.emit_span_lint(NON_LOCAL_DEFINITIONS, ms, NonLocalDefinitionsDiag::Impl {
                    depth: self.body_depth,
                    body_kind_descr: cx.tcx.def_kind_descr(parent_def_kind, parent),
                    body_name: parent_opt_item_name
                        .map(|s| s.to_ident_string())
                        .unwrap_or_else(|| "<unnameable>".to_string()),
                    cargo_update: cargo_update(),
                    const_anon,
                    doctest,
                    macro_to_change,
                })
            }
            ItemKind::Macro(_macro, MacroKind::Bang)
                if cx.tcx.has_attr(item.owner_id.def_id, sym::macro_export) =>
            {
                cx.emit_span_lint(
                    NON_LOCAL_DEFINITIONS,
                    item.span,
                    NonLocalDefinitionsDiag::MacroRules {
                        depth: self.body_depth,
                        body_kind_descr: cx.tcx.def_kind_descr(parent_def_kind, parent),
                        body_name: parent_opt_item_name
                            .map(|s| s.to_ident_string())
                            .unwrap_or_else(|| "<unnameable>".to_string()),
                        cargo_update: cargo_update(),
                        doctest: is_at_toplevel_doctest(),
                    },
                )
            }
            _ => {}
        }
    }
}

/// Simple hir::Path collector
struct PathCollector<'tcx> {
    paths: Vec<Path<'tcx>>,
}

impl<'tcx> Visitor<'tcx> for PathCollector<'tcx> {
    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        self.paths.push(path.clone()); // need to clone, bc of the restricted lifetime
        intravisit::walk_path(self, path)
    }
}

/// Given a path and a parent impl def id, this checks if the if parent resolution
/// def id correspond to the def id of the parent impl definition.
///
/// Given this path, we will look at the path (and ignore any generic args):
///
/// ```text
///    std::convert::PartialEq<Foo<Bar>>
///    ^^^^^^^^^^^^^^^^^^^^^^^
/// ```
#[inline]
fn path_has_local_parent(
    path: &Path<'_>,
    cx: &LateContext<'_>,
    impl_parent: DefId,
    impl_parent_parent: Option<DefId>,
) -> bool {
    path.res
        .opt_def_id()
        .is_some_and(|did| did_has_local_parent(did, cx.tcx, impl_parent, impl_parent_parent))
}

/// Given a def id and a parent impl def id, this checks if the parent
/// def id (modulo modules) correspond to the def id of the parent impl definition.
#[inline]
fn did_has_local_parent(
    did: DefId,
    tcx: TyCtxt<'_>,
    impl_parent: DefId,
    impl_parent_parent: Option<DefId>,
) -> bool {
    did.is_local()
        && if let Some(did_parent) = tcx.opt_parent(did) {
            did_parent == impl_parent
                || Some(did_parent) == impl_parent_parent
                || !did_parent.is_crate_root()
                    && tcx.def_kind(did_parent) == DefKind::Mod
                    && did_has_local_parent(did_parent, tcx, impl_parent, impl_parent_parent)
        } else {
            false
        }
}

/// Return for a given `Path` the span until the last args
fn path_span_without_args(path: &Path<'_>) -> Span {
    if let Some(args) = &path.segments.last().unwrap().args {
        path.span.until(args.span_ext)
    } else {
        path.span
    }
}

/// Return a "error message-able" ident for the last segment of the `Path`
fn path_name_to_string(path: &Path<'_>) -> String {
    path.segments.last().unwrap().ident.name.to_ident_string()
}
