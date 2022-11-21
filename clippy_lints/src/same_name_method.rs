use clippy_utils::diagnostics::span_lint_hir_and_then;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{HirId, Impl, ItemKind, Node, Path, QPath, TraitRef, TyKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::AssocKind;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Symbol;
use rustc_span::Span;
use std::collections::{BTreeMap, BTreeSet};

declare_clippy_lint! {
    /// ### What it does
    /// It lints if a struct has two methods with the same name:
    /// one from a trait, another not from trait.
    ///
    /// ### Why is this bad?
    /// Confusing.
    ///
    /// ### Example
    /// ```rust
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
    #[expect(clippy::too_many_lines)]
    fn check_crate_post(&mut self, cx: &LateContext<'tcx>) {
        let mut map = FxHashMap::<Res, ExistingName>::default();

        for id in cx.tcx.hir().items() {
            if matches!(cx.tcx.def_kind(id.owner_id), DefKind::Impl)
                && let item = cx.tcx.hir().item(id)
                && let ItemKind::Impl(Impl {
                    items,
                    of_trait,
                    self_ty,
                    ..
                }) = &item.kind
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
                    Some(trait_ref) => {
                        let mut methods_in_trait: BTreeSet<Symbol> = if_chain! {
                            if let Some(Node::TraitRef(TraitRef { path, .. })) =
                                cx.tcx.hir().find(trait_ref.hir_ref_id);
                            if let Res::Def(DefKind::Trait, did) = path.res;
                            then{
                                // FIXME: if
                                // `rustc_middle::ty::assoc::AssocItems::items` is public,
                                // we can iterate its keys instead of `in_definition_order`,
                                // which's more efficient
                                cx.tcx
                                    .associated_items(did)
                                    .in_definition_order()
                                    .filter(|assoc_item| {
                                        matches!(assoc_item.kind, AssocKind::Fn)
                                    })
                                    .map(|assoc_item| assoc_item.name)
                                    .collect()
                            }else{
                                BTreeSet::new()
                            }
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
                                            &format!("existing `{method_name}` defined here"),
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

                        for impl_item_ref in (*items).iter().filter(|impl_item_ref| {
                            matches!(impl_item_ref.kind, rustc_hir::AssocItemKind::Fn { .. })
                        }) {
                            let method_name = impl_item_ref.ident.name;
                            methods_in_trait.remove(&method_name);
                            check_trait_method(method_name, impl_item_ref.span);
                        }

                        for method_name in methods_in_trait {
                            check_trait_method(method_name, item.span);
                        }
                    },
                    None => {
                        for impl_item_ref in (*items).iter().filter(|impl_item_ref| {
                            matches!(impl_item_ref.kind, rustc_hir::AssocItemKind::Fn { .. })
                        }) {
                            let method_name = impl_item_ref.ident.name;
                            let impl_span = impl_item_ref.span;
                            let hir_id = impl_item_ref.id.hir_id();
                            if let Some(trait_spans) = existing_name.trait_methods.get(&method_name) {
                                span_lint_hir_and_then(
                                    cx,
                                    SAME_NAME_METHOD,
                                    hir_id,
                                    impl_span,
                                    "method's name is the same as an existing method in a trait",
                                    |diag| {
                                        // TODO should we `span_note` on every trait?
                                        // iterate on trait_spans?
                                        diag.span_note(
                                            trait_spans[0],
                                            &format!("existing `{method_name}` defined here"),
                                        );
                                    },
                                );
                            }
                            existing_name.impl_methods.insert(method_name, (impl_span, hir_id));
                        }
                    },
                }
            }
        }
    }
}
