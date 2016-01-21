use rustc::lint::*;
use rustc_front::hir::*;
use syntax::ast::{Attribute, MetaItem_};
use utils::{match_path, span_lint_and_then};
use utils::HASH_PATH;

use rustc::middle::ty::fast_reject::simplify_type;

/// **What it does:** This lint warns about deriving `Hash` but implementing `PartialEq`
/// explicitely.
///
/// **Why is this bad?** The implementation of these traits must agree (for example for use with
/// `HashMap`) so it’s probably a bad idea to use a default-generated `Hash` implementation  with
/// an explicitely defined `PartialEq`. In particular, the following must hold for any type:
///
/// ```rust
/// k1 == k2 -> hash(k1) == hash(k2)
/// ```
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// #[derive(Hash)]
/// struct Foo;
///
/// impl PartialEq for Foo {
///     ..
/// }
declare_lint! {
    pub DERIVE_HASH_NOT_EQ,
    Warn,
    "deriving `Hash` but implementing `PartialEq` explicitly"
}

pub struct Derive;

impl LintPass for Derive {
    fn get_lints(&self) -> LintArray {
        lint_array!(DERIVE_HASH_NOT_EQ)
    }
}

impl LateLintPass for Derive {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        /// A `#[derive]`d implementation has a `#[automatically_derived]` attribute.
        fn is_automatically_derived(attr: &Attribute) -> bool {
            if let MetaItem_::MetaWord(ref word) = attr.node.value.node {
                word == &"automatically_derived"
            }
            else {
                false
            }
        }

        // If `item` is an automatically derived `Hash` implementation
        if_let_chain! {[
            let ItemImpl(_, _, _, Some(ref trait_ref), ref ast_ty, _) = item.node,
            match_path(&trait_ref.path, &HASH_PATH),
            item.attrs.iter().any(is_automatically_derived),
            let Some(peq_trait_def_id) = cx.tcx.lang_items.eq_trait()
        ], {
            let peq_trait_def = cx.tcx.lookup_trait_def(peq_trait_def_id);

            cx.tcx.populate_implementations_for_trait_if_necessary(peq_trait_def.trait_ref.def_id);
            let peq_impls = peq_trait_def.borrow_impl_lists(cx.tcx).1;
            let ast_ty_to_ty_cache = cx.tcx.ast_ty_to_ty_cache.borrow();


            // Look for the PartialEq implementations for `ty`
            if_let_chain! {[
                let Some(ty) = ast_ty_to_ty_cache.get(&ast_ty.id),
                let Some(simpl_ty) = simplify_type(cx.tcx, ty, false),
                let Some(impl_ids) = peq_impls.get(&simpl_ty)
            ], {
                for &impl_id in impl_ids {
                    let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                    // Only care about `impl PartialEq<Foo> for Foo`
                    if trait_ref.input_types()[0] == *ty &&
                      !cx.tcx.get_attrs(impl_id).iter().any(is_automatically_derived) {
                        span_lint_and_then(
                            cx, DERIVE_HASH_NOT_EQ, item.span,
                            &format!("you are deriving `Hash` but have implemented \
                                      `PartialEq` explicitely"), |db| {
                            if let Some(node_id) = cx.tcx.map.as_local_node_id(impl_id) {
                                db.span_note(
                                    cx.tcx.map.span(node_id),
                                    "`PartialEq` implemented here"
                                );
                            }
                        });
                    }
                }
            }}
        }}
    }
}
