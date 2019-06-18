use crate::utils::paths;
use crate::utils::sugg::DiagnosticBuilderExt;
use crate::utils::{get_trait_def_id, implements_trait, return_ty, same_tys, span_lint_hir_and_then};
use if_chain::if_chain;
use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::ty::{self, Ty};
use rustc::util::nodemap::NodeSet;
use rustc::{declare_tool_lint, impl_lint_pass};
use rustc_errors::Applicability;
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for types with a `fn new() -> Self` method and no
    /// implementation of
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html).
    ///
    /// It detects both the case when a manual
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
    /// implementation is required and also when it can be created with
    /// `#[derive(Default)]`
    ///
    /// **Why is this bad?** The user might expect to be able to use
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html) as the
    /// type can be constructed without arguments.
    ///
    /// **Known problems:** Hopefully none.
    ///
    /// **Example:**
    ///
    /// ```ignore
    /// struct Foo(Bar);
    ///
    /// impl Foo {
    ///     fn new() -> Self {
    ///         Foo(Bar::new())
    ///     }
    /// }
    /// ```
    ///
    /// Instead, use:
    ///
    /// ```ignore
    /// struct Foo(Bar);
    ///
    /// impl Default for Foo {
    ///     fn default() -> Self {
    ///         Foo(Bar::new())
    ///     }
    /// }
    /// ```
    ///
    /// Or, if
    /// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
    /// can be derived by `#[derive(Default)]`:
    ///
    /// ```rust
    /// struct Foo;
    ///
    /// impl Foo {
    ///     fn new() -> Self {
    ///         Foo
    ///     }
    /// }
    /// ```
    ///
    /// Instead, use:
    ///
    /// ```rust
    /// #[derive(Default)]
    /// struct Foo;
    ///
    /// impl Foo {
    ///     fn new() -> Self {
    ///         Foo
    ///     }
    /// }
    /// ```
    ///
    /// You can also have `new()` call `Default::default()`.
    pub NEW_WITHOUT_DEFAULT,
    style,
    "`fn new() -> Self` method without `Default` implementation"
}

#[derive(Clone, Default)]
pub struct NewWithoutDefault {
    impling_types: Option<NodeSet>,
}

impl_lint_pass!(NewWithoutDefault => [NEW_WITHOUT_DEFAULT]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NewWithoutDefault {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::Item) {
        if let hir::ItemKind::Impl(_, _, _, _, None, _, ref items) = item.node {
            for assoc_item in items {
                if let hir::AssocItemKind::Method { has_self: false } = assoc_item.kind {
                    let impl_item = cx.tcx.hir().impl_item(assoc_item.id);
                    if in_external_macro(cx.sess(), impl_item.span) {
                        return;
                    }
                    if let hir::ImplItemKind::Method(ref sig, _) = impl_item.node {
                        let name = impl_item.ident.name;
                        let id = impl_item.hir_id;
                        if sig.header.constness == hir::Constness::Const {
                            // can't be implemented by default
                            return;
                        }
                        if sig.header.unsafety == hir::Unsafety::Unsafe {
                            // can't be implemented for unsafe new
                            return;
                        }
                        if impl_item.generics.params.iter().any(|gen| match gen.kind {
                            hir::GenericParamKind::Type { .. } => true,
                            _ => false,
                        }) {
                            // when the result of `new()` depends on a type parameter we should not require
                            // an
                            // impl of `Default`
                            return;
                        }
                        if sig.decl.inputs.is_empty() && name == sym!(new) && cx.access_levels.is_reachable(id) {
                            let self_did = cx.tcx.hir().local_def_id_from_hir_id(cx.tcx.hir().get_parent_item(id));
                            let self_ty = cx.tcx.type_of(self_did);
                            if_chain! {
                                if same_tys(cx, self_ty, return_ty(cx, id));
                                if let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT);
                                then {
                                    if self.impling_types.is_none() {
                                        let mut impls = NodeSet::default();
                                        cx.tcx.for_each_impl(default_trait_id, |d| {
                                            if let Some(ty_def) = cx.tcx.type_of(d).ty_adt_def() {
                                                if let Some(node_id) = cx.tcx.hir().as_local_node_id(ty_def.did) {
                                                    impls.insert(node_id);
                                                }
                                            }
                                        });
                                        self.impling_types = Some(impls);
                                    }

                                    // Check if a Default implementation exists for the Self type, regardless of
                                    // generics
                                    if_chain! {
                                        if let Some(ref impling_types) = self.impling_types;
                                        if let Some(self_def) = cx.tcx.type_of(self_did).ty_adt_def();
                                        if self_def.did.is_local();
                                        then {
                                            let self_id = cx.tcx.hir().local_def_id_to_hir_id(self_def.did.to_local());
                                            let node_id = cx.tcx.hir().hir_to_node_id(self_id);
                                            if impling_types.contains(&node_id) {
                                                return;
                                            }
                                        }
                                    }

                                    if let Some(sp) = can_derive_default(self_ty, cx, default_trait_id) {
                                        span_lint_hir_and_then(
                                            cx,
                                            NEW_WITHOUT_DEFAULT,
                                            id,
                                            impl_item.span,
                                            &format!(
                                                "you should consider deriving a `Default` implementation for `{}`",
                                                self_ty
                                            ),
                                            |db| {
                                                db.suggest_item_with_attr(
                                                    cx,
                                                    sp,
                                                    "try this",
                                                    "#[derive(Default)]",
                                                    Applicability::MaybeIncorrect,
                                                );
                                            });
                                    } else {
                                        span_lint_hir_and_then(
                                            cx,
                                            NEW_WITHOUT_DEFAULT,
                                            id,
                                            impl_item.span,
                                            &format!(
                                                "you should consider adding a `Default` implementation for `{}`",
                                                self_ty
                                            ),
                                            |db| {
                                                db.suggest_prepend_item(
                                                    cx,
                                                    item.span,
                                                    "try this",
                                                    &create_new_without_default_suggest_msg(self_ty),
                                                    Applicability::MaybeIncorrect,
                                                );
                                            },
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn create_new_without_default_suggest_msg(ty: Ty<'_>) -> String {
    #[rustfmt::skip]
    format!(
"impl Default for {} {{
    fn default() -> Self {{
        Self::new()
    }}
}}", ty)
}

fn can_derive_default<'t, 'c>(ty: Ty<'t>, cx: &LateContext<'c, 't>, default_trait_id: DefId) -> Option<Span> {
    match ty.sty {
        ty::Adt(adt_def, substs) if adt_def.is_struct() => {
            for field in adt_def.all_fields() {
                let f_ty = field.ty(cx.tcx, substs);
                if !implements_trait(cx, f_ty, default_trait_id, &[]) {
                    return None;
                }
            }
            Some(cx.tcx.def_span(adt_def.did))
        },
        _ => None,
    }
}
