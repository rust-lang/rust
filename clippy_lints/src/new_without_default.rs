use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::lint::*;
use rustc::ty::{self, Ty};
use syntax::codemap::Span;
use utils::paths;
use utils::{get_trait_def_id, implements_trait, in_external_macro, return_ty, same_tys, span_lint_and_then};
use utils::sugg::DiagnosticBuilderExt;

/// **What it does:** Checks for types with a `fn new() -> Self` method and no
/// implementation of
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html).
///
/// **Why is this bad?** The user might expect to be able to use
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html) as the
/// type can be constructed without arguments.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
///
/// ```rust,ignore
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
/// ```rust
/// struct Foo(Bar);
///
/// impl Default for Foo {
///     fn default() -> Self {
///         Foo(Bar::new())
///     }
/// }
/// ```
///
/// You can also have `new()` call `Default::default()`.
declare_lint! {
    pub NEW_WITHOUT_DEFAULT,
    Warn,
    "`fn new() -> Self` method without `Default` implementation"
}

/// **What it does:** Checks for types with a `fn new() -> Self` method
/// and no implementation of
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html),
/// where the `Default` can be derived by `#[derive(Default)]`.
///
/// **Why is this bad?** The user might expect to be able to use
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html) as the
/// type can be constructed without arguments.
///
/// **Known problems:** Hopefully none.
///
/// **Example:**
///
/// ```rust,ignore
/// struct Foo;
///
/// impl Foo {
///     fn new() -> Self {
///         Foo
///     }
/// }
/// ```
///
/// Just prepend `#[derive(Default)]` before the `struct` definition.
declare_lint! {
    pub NEW_WITHOUT_DEFAULT_DERIVE,
    Warn,
    "`fn new() -> Self` without `#[derive]`able `Default` implementation"
}

#[derive(Copy, Clone)]
pub struct NewWithoutDefault;

impl LintPass for NewWithoutDefault {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEW_WITHOUT_DEFAULT, NEW_WITHOUT_DEFAULT_DERIVE)
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NewWithoutDefault {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx hir::Item) {
        if let hir::ItemImpl(_, _, _, _, None, _, ref items) = item.node {
            for assoc_item in items {
                if let hir::AssociatedItemKind::Method { has_self: false } = assoc_item.kind {
                    let impl_item = cx.tcx.hir.impl_item(assoc_item.id);
                    if in_external_macro(cx, impl_item.span) {
                        return;
                    }
                    if let hir::ImplItemKind::Method(ref sig, _) = impl_item.node {
                        let name = impl_item.name;
                        let id = impl_item.id;
                        if sig.constness == hir::Constness::Const {
                            // can't be implemented by default
                            return;
                        }
                        if impl_item.generics.params.iter().any(|gen| gen.is_type_param()) {
                            // when the result of `new()` depends on a type parameter we should not require
                            // an
                            // impl of `Default`
                            return;
                        }
                        if sig.decl.inputs.is_empty() && name == "new" && cx.access_levels.is_reachable(id) {
                            let self_ty = cx.tcx
                                .type_of(cx.tcx.hir.local_def_id(cx.tcx.hir.get_parent(id)));
                            if_chain! {
                                if same_tys(cx, self_ty, return_ty(cx, id));
                                if let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT);
                                if !implements_trait(cx, self_ty, default_trait_id, &[]);
                                then {
                                    if let Some(sp) = can_derive_default(self_ty, cx, default_trait_id) {
                                        span_lint_and_then(
                                            cx,
                                            NEW_WITHOUT_DEFAULT_DERIVE,
                                            impl_item.span,
                                            &format!("you should consider deriving a `Default` implementation for `{}`", self_ty),
                                            |db| {
                                                db.suggest_item_with_attr(cx, sp, "try this", "#[derive(Default)]");
                                            });
                                    } else {
                                        span_lint_and_then(
                                            cx,
                                            NEW_WITHOUT_DEFAULT,
                                            impl_item.span,
                                            &format!("you should consider adding a `Default` implementation for `{}`", self_ty),
                                            |db| {
                                                db.suggest_prepend_item(
                                                    cx,
                                                    item.span,
                                                    "try this",
                                                    &create_new_without_default_suggest_msg(self_ty),
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

fn create_new_without_default_suggest_msg(ty: Ty) -> String {
    #[rustfmt_skip]
    format!(
"impl Default for {} {{
    fn default() -> Self {{
        Self::new()
    }}
}}", ty)
}

fn can_derive_default<'t, 'c>(ty: Ty<'t>, cx: &LateContext<'c, 't>, default_trait_id: DefId) -> Option<Span> {
    match ty.sty {
        ty::TyAdt(adt_def, substs) if adt_def.is_struct() => {
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
