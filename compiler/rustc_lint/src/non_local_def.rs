use rustc_hir::{def::DefKind, Body, Item, ItemKind, Node, Path, QPath, TyKind};
use rustc_span::def_id::{DefId, LOCAL_CRATE};
use rustc_span::{sym, symbol::kw, ExpnKind, MacroKind};

use smallvec::{smallvec, SmallVec};

use crate::lints::{NonLocalDefinitionsCargoUpdateNote, NonLocalDefinitionsDiag};
use crate::{LateContext, LateLintPass, LintContext};

declare_lint! {
    /// The `non_local_definitions` lint checks for `impl` blocks and `#[macro_export]`
    /// macro inside bodies (functions, enum discriminant, ...).
    ///
    /// ### Example
    ///
    /// ```rust
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
    /// and higher, see see the tracking issue <https://github.com/rust-lang/rust/issues/120363>.
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
pub struct NonLocalDefinitions {
    body_depth: u32,
}

impl_lint_pass!(NonLocalDefinitions => [NON_LOCAL_DEFINITIONS]);

// FIXME(Urgau): Figure out how to handle modules nested in bodies.
// It's currently not handled by the current logic because modules are not bodies.
// They don't even follow the correct order (check_body -> check_mod -> check_body_post)
// instead check_mod is called after every body has been handled.

impl<'tcx> LateLintPass<'tcx> for NonLocalDefinitions {
    fn check_body(&mut self, _cx: &LateContext<'tcx>, _body: &'tcx Body<'tcx>) {
        self.body_depth += 1;
    }

    fn check_body_post(&mut self, _cx: &LateContext<'tcx>, _body: &'tcx Body<'tcx>) {
        self.body_depth -= 1;
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'tcx>) {
        if self.body_depth == 0 {
            return;
        }

        let parent = cx.tcx.parent(item.owner_id.def_id.into());
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
                && std::env::var_os("CARGO").is_some()
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
                // _Type_, and we look inside thoses paths to try a find in one
                // of them a type whose parent is the same as the impl definition.
                //
                // If that's the case this means that this impl block declaration
                // is using local items and so we don't lint on it.

                // We also ignore anon-const in item by including the anon-const
                // parent as well; and since it's quite uncommon, we use smallvec
                // to avoid unnecessary heap allocations.
                let local_parents: SmallVec<[DefId; 1]> = if parent_def_kind == DefKind::Const
                    && parent_opt_item_name == Some(kw::Underscore)
                {
                    smallvec![parent, cx.tcx.parent(parent)]
                } else {
                    smallvec![parent]
                };

                let self_ty_has_local_parent = match impl_.self_ty.kind {
                    TyKind::Path(QPath::Resolved(_, ty_path)) => {
                        path_has_local_parent(ty_path, cx, &*local_parents)
                    }
                    TyKind::TraitObject([principle_poly_trait_ref, ..], _, _) => {
                        path_has_local_parent(
                            principle_poly_trait_ref.trait_ref.path,
                            cx,
                            &*local_parents,
                        )
                    }
                    TyKind::TraitObject([], _, _)
                    | TyKind::InferDelegation(_, _)
                    | TyKind::Slice(_)
                    | TyKind::Array(_, _)
                    | TyKind::Ptr(_)
                    | TyKind::Ref(_, _)
                    | TyKind::BareFn(_)
                    | TyKind::Never
                    | TyKind::Tup(_)
                    | TyKind::Path(_)
                    | TyKind::AnonAdt(_)
                    | TyKind::OpaqueDef(_, _, _)
                    | TyKind::Typeof(_)
                    | TyKind::Infer
                    | TyKind::Err(_) => false,
                };

                let of_trait_has_local_parent = impl_
                    .of_trait
                    .map(|of_trait| path_has_local_parent(of_trait.path, cx, &*local_parents))
                    .unwrap_or(false);

                // If none of them have a local parent (LOGICAL NOR) this means that
                // this impl definition is a non-local definition and so we lint on it.
                if !(self_ty_has_local_parent || of_trait_has_local_parent) {
                    let const_anon = if self.body_depth == 1
                        && parent_def_kind == DefKind::Const
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

                    cx.emit_span_lint(
                        NON_LOCAL_DEFINITIONS,
                        item.span,
                        NonLocalDefinitionsDiag::Impl {
                            depth: self.body_depth,
                            body_kind_descr: cx.tcx.def_kind_descr(parent_def_kind, parent),
                            body_name: parent_opt_item_name
                                .map(|s| s.to_ident_string())
                                .unwrap_or_else(|| "<unnameable>".to_string()),
                            cargo_update: cargo_update(),
                            const_anon,
                        },
                    )
                }
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
                    },
                )
            }
            _ => {}
        }
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
fn path_has_local_parent(path: &Path<'_>, cx: &LateContext<'_>, local_parents: &[DefId]) -> bool {
    path.res.opt_def_id().is_some_and(|did| local_parents.contains(&cx.tcx.parent(did)))
}
