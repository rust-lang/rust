use rustc::hir::intravisit::FnKind;
use rustc::hir::def_id::DefId;
use rustc::hir;
use rustc::lint::*;
use rustc::ty;
use syntax::ast;
use syntax::codemap::Span;
use utils::paths;
use utils::{get_trait_def_id, implements_trait, in_external_macro, return_ty, same_tys, span_lint};

/// **What it does:** This lints about type with a `fn new() -> Self` method
/// and no implementation of
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
///
/// **Why is this bad?** User might expect to be able to use
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
/// as the type can be
/// constructed without arguments.
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
/// You can also have `new()` call `Default::default()`
declare_lint! {
    pub NEW_WITHOUT_DEFAULT,
    Warn,
    "`fn new() -> Self` method without `Default` implementation"
}

/// **What it does:** This lints about type with a `fn new() -> Self` method
/// and no implementation of
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
///
/// **Why is this bad?** User might expect to be able to use
/// [`Default`](https://doc.rust-lang.org/std/default/trait.Default.html)
/// as the type can be
/// constructed without arguments.
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
/// Just prepend `#[derive(Default)]` before the `struct` definition
declare_lint! {
    pub NEW_WITHOUT_DEFAULT_DERIVE,
    Warn,
    "`fn new() -> Self` without `#[derive]`able `Default` implementation"
}

#[derive(Copy,Clone)]
pub struct NewWithoutDefault;

impl LintPass for NewWithoutDefault {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEW_WITHOUT_DEFAULT, NEW_WITHOUT_DEFAULT_DERIVE)
    }
}

impl LateLintPass for NewWithoutDefault {
    fn check_fn(&mut self, cx: &LateContext, kind: FnKind, decl: &hir::FnDecl, _: &hir::Block, span: Span, id: ast::NodeId) {
        if in_external_macro(cx, span) {
            return;
        }

        if let FnKind::Method(name, ref sig, _, _) = kind {
            if sig.constness == hir::Constness::Const {
                // can't be implemented by default
                return;
            }
            if decl.inputs.is_empty() && name.as_str() == "new" && cx.access_levels.is_reachable(id) {
                let self_ty = cx.tcx
                    .lookup_item_type(cx.tcx.map.local_def_id(cx.tcx.map.get_parent(id)))
                    .ty;
                if_let_chain!{[
                    self_ty.walk_shallow().next().is_none(), // implements_trait does not work with generics
                    let Some(ret_ty) = return_ty(cx, id),
                    same_tys(cx, self_ty, ret_ty, id),
                    let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT),
                    !implements_trait(cx, self_ty, default_trait_id, Vec::new())
                ], {
                    if can_derive_default(self_ty, cx, default_trait_id) {
                        span_lint(cx,
                                  NEW_WITHOUT_DEFAULT_DERIVE, span,
                                  &format!("you should consider deriving a \
                                           `Default` implementation for `{}`",
                                           self_ty)).
                                  span_suggestion(span,
                                                  "try this",
                                                  "#[derive(Default)]".into());
                    } else {
                        span_lint(cx,
                                  NEW_WITHOUT_DEFAULT, span,
                                  &format!("you should consider adding a \
                                           `Default` implementation for `{}`",
                                           self_ty)).
                                  span_suggestion(span,
                                                  "try this",
                             format!("impl Default for {} {{ fn default() -> \
                                    Self {{ {}::new() }} }}", self_ty, self_ty));
                    }
                }}
            }
        }
    }
}

fn can_derive_default<'t, 'c>(ty: ty::Ty<'t>, cx: &LateContext<'c, 't>, default_trait_id: DefId) -> bool {
    match ty.sty {
        ty::TyStruct(ref adt_def, ref substs) => {
            for field in adt_def.all_fields() {
                let f_ty = field.ty(cx.tcx, substs);
                if !implements_trait(cx, f_ty, default_trait_id, Vec::new()) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}
