use clippy_utils::diagnostics::span_lint_hir_and_then;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{HirId, Impl, ItemKind, Node, Path, QPath, TraitRef, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{AssocItem, AssocKind};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::Symbol;
use std::collections::{BTreeMap, BTreeSet};

declare_clippy_lint! {
    /// ### What it does
    /// It lints if a struct has two methods with the same name:
    /// one from a trait, another not from a trait.
    ///
    /// ### Why restrict this?
    /// Confusing.
    ///
    /// ### Example
    /// ```no_run
    /// trait T {
    ///     fn foo(&self) {}
    /// }
    ///
    /// struct S;
    ///
    /// impl T for S {
    ///     fn foo(&self) {}
    /// }
    ///
    /// impl S {
    ///     fn foo(&self) {}
    /// }
    /// ```
    #[clippy::version = "1.57.0"]
    pub SAME_NAME_METHOD,
    restriction,
    "two method with same name"
}

declare_lint_pass!(SameNameMethod => [SAME_NAME_METHOD]);

struct ExistingName {
    impl_methods: BTreeMap<Symbol, (Span, HirId)>,
    trait_methods: BTreeMap<Symbol, Vec<Span>>,
}

impl<'tcx> LateLintPass<'tcx> for SameNameMethod {
    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let mut map = FxHashMap::<Res, ExistingName>::default();

        for id in cx.tcx.hir_free_items() {
            if matches!(cx.tcx.def_kind(id.owner_id), DefKind::Impl { .. })
                && let item = cx.tcx.hir_item(id)
                && let ItemKind::Impl(Impl { of_trait, self_ty, .. }) = &item.kind
                && let TyKind::Path(QPath::Resolved(_, Path { res, .. })) = self_ty.kind
            {
                if !map.contains_key(res) {
                    map.insert(
                        *res,
                        ExistingName {
                            impl_methods: BTreeMap::new(),
                            trait_methods: BTreeMap::new(),
                        },
                    );
                }
                let existing_name = map.get_mut(res).unwrap();

                match of_trait {
                    Some(of_trait) => {
                        let mut methods_in_trait: BTreeSet<Symbol> = if let Node::TraitRef(TraitRef { path, .. }) =
                            cx.tcx.hir_node(of_trait.trait_ref.hir_ref_id)
                            && let Res::Def(DefKind::Trait, did) = path.res
                        {
                            // FIXME: if
                            // `rustc_middle::ty::assoc::AssocItems::items` is public,
                            // we can iterate its keys instead of `in_definition_order`,
                            // which's more efficient
                            cx.tcx
                                .associated_items(did)
                                .in_definition_order()
                                .filter(|assoc_item| assoc_item.is_fn())
                                .map(AssocItem::name)
                                .collect()
                        } else {
                            BTreeSet::new()
                        };

                        let mut check_trait_method = |method_name: Symbol, trait_method_span: Span| {
                            if let Some((impl_span, hir_id)) = existing_name.impl_methods.get(&method_name) {
                                span_lint_hir_and_then(
                                    cx,
                                    SAME_NAME_METHOD,
                                    *hir_id,
                                    *impl_span,
                                    "method's name is the same as an existing method in a trait",
                                    |diag| {
                                        diag.span_note(
                                            trait_method_span,
                                            format!("existing `{method_name}` defined here"),
                                        );
                                    },
                                );
                            }
                            if let Some(v) = existing_name.trait_methods.get_mut(&method_name) {
                                v.push(trait_method_span);
                            } else {
                                existing_name.trait_methods.insert(method_name, vec![trait_method_span]);
                            }
                        };

                        for assoc_item in cx.tcx.associated_items(id.owner_id).in_definition_order() {
                            if let AssocKind::Fn { name, .. } = assoc_item.kind {
                                methods_in_trait.remove(&name);
                                check_trait_method(name, cx.tcx.def_span(assoc_item.def_id));
                            }
                        }

                        for method_name in methods_in_trait {
                            check_trait_method(method_name, item.span);
                        }
                    },
                    None => {
                        for assoc_item in cx.tcx.associated_items(id.owner_id).in_definition_order() {
                            let AssocKind::Fn { name, .. } = assoc_item.kind else {
                                continue;
                            };
                            let impl_span = cx.tcx.def_span(assoc_item.def_id);
                            let hir_id = cx.tcx.local_def_id_to_hir_id(assoc_item.def_id.expect_local());
                            if let Some(trait_spans) = existing_name.trait_methods.get(&name) {
                                span_lint_hir_and_then(
                                    cx,
                                    SAME_NAME_METHOD,
                                    hir_id,
                                    impl_span,
                                    "method's name is the same as an existing method in a trait",
                                    |diag| {
                                        // TODO should we `span_note` on every trait?
                                        // iterate on trait_spans?
                                        diag.span_note(trait_spans[0], format!("existing `{name}` defined here"));
                                    },
                                );
                            }
                            existing_name.impl_methods.insert(name, (impl_span, hir_id));
                        }
                    },
                }
            }
        }
    }
}
