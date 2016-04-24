use rustc::hir::intravisit::FnKind;
use rustc::hir;
use rustc::lint::*;
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
/// struct Foo;
///
/// impl Default for Foo {
///     fn default() -> Self {
///         Foo
///     }
/// }
/// ```
///
/// You can also have `new()` call `Default::default()`
///
declare_lint! {
    pub NEW_WITHOUT_DEFAULT,
    Warn,
    "`fn new() -> Self` method without `Default` implementation"
}

#[derive(Copy,Clone)]
pub struct NewWithoutDefault;

impl LintPass for NewWithoutDefault {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEW_WITHOUT_DEFAULT)
    }
}

impl LateLintPass for NewWithoutDefault {
    fn check_fn(&mut self, cx: &LateContext, kind: FnKind, decl: &hir::FnDecl, _: &hir::Block, span: Span, id: ast::NodeId) {
        if in_external_macro(cx, span) {
            return;
        }

        if let FnKind::Method(name, _, _, _) = kind {
            if decl.inputs.is_empty() && name.as_str() == "new" {
                let self_ty = cx.tcx.lookup_item_type(cx.tcx.map.local_def_id(cx.tcx.map.get_parent(id))).ty;

                if_let_chain!{[
                    self_ty.walk_shallow().next().is_none(), // implements_trait does not work with generics
                    let Some(ret_ty) = return_ty(cx, id),
                    same_tys(cx, self_ty, ret_ty, id),
                    let Some(default_trait_id) = get_trait_def_id(cx, &paths::DEFAULT_TRAIT),
                    !implements_trait(cx, self_ty, default_trait_id, Vec::new())
                ], {
                    span_lint(cx, NEW_WITHOUT_DEFAULT, span,
                              &format!("you should consider adding a `Default` implementation for `{}`", self_ty));
                }}
            }
        }
    }
}
