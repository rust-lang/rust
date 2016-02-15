use rustc::lint::*;
use rustc::middle::subst::Subst;
use rustc::middle::ty::TypeVariants;
use rustc::middle::ty::fast_reject::simplify_type;
use rustc::middle::ty;
use rustc_front::hir::*;
use syntax::ast::{Attribute, MetaItemKind};
use syntax::codemap::Span;
use utils::{CLONE_TRAIT_PATH, HASH_PATH};
use utils::{match_path, span_lint_and_then};

/// **What it does:** This lint warns about deriving `Hash` but implementing `PartialEq`
/// explicitly.
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
/// ```
declare_lint! {
    pub DERIVE_HASH_NOT_EQ,
    Warn,
    "deriving `Hash` but implementing `PartialEq` explicitly"
}

/// **What it does:** This lint warns about explicit `Clone` implementation for `Copy` types.
///
/// **Why is this bad?** To avoid surprising behaviour, these traits should agree and the behaviour
/// of `Copy` cannot be overridden. In almost all situations a `Copy` type should have a `Clone`
/// implementation that does nothing more than copy the object, which is what
/// `#[derive(Copy, Clone)]` gets you.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// #[derive(Copy)]
/// struct Foo;
///
/// impl Clone for Foo {
///     ..
/// }
/// ```
declare_lint! {
    pub EXPL_IMPL_CLONE_ON_COPY,
    Warn,
    "implementing `Clone` explicitly on `Copy` types"
}

pub struct Derive;

impl LintPass for Derive {
    fn get_lints(&self) -> LintArray {
        lint_array!(EXPL_IMPL_CLONE_ON_COPY, DERIVE_HASH_NOT_EQ)
    }
}

impl LateLintPass for Derive {
    fn check_item(&mut self, cx: &LateContext, item: &Item) {
        

        if_let_chain! {[
            let ItemImpl(_, _, _, Some(ref trait_ref), _, _) = item.node
        ], {

            let ty = cx.tcx.lookup_item_type(cx.tcx.map.local_def_id(item.id)).ty;
            if item.attrs.iter().any(is_automatically_derived) {
                check_hash_peq(cx, item.span, trait_ref, ty);
            }
            else {
                check_copy_clone(cx, item, trait_ref, ty);
            }
        }}
    }
}

/// Implementation of the `DERIVE_HASH_NOT_EQ` lint.
fn check_hash_peq(cx: &LateContext, span: Span, trait_ref: &TraitRef, ty: ty::Ty) {
    // If `item` is an automatically derived `Hash` implementation
    if_let_chain! {[
        match_path(&trait_ref.path, &HASH_PATH),
        let Some(peq_trait_def_id) = cx.tcx.lang_items.eq_trait()
    ], {
        let peq_trait_def = cx.tcx.lookup_trait_def(peq_trait_def_id);

        cx.tcx.populate_implementations_for_trait_if_necessary(peq_trait_def.trait_ref.def_id);
        let peq_impls = peq_trait_def.borrow_impl_lists(cx.tcx).1;

        // Look for the PartialEq implementations for `ty`
        if_let_chain! {[
            let Some(simpl_ty) = simplify_type(cx.tcx, ty, false),
            let Some(impl_ids) = peq_impls.get(&simpl_ty)
        ], {
            for &impl_id in impl_ids {
                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialEq<Foo> for Foo`
                if trait_ref.input_types()[0] == ty &&
                  !cx.tcx.get_attrs(impl_id).iter().any(is_automatically_derived) {
                    span_lint_and_then(
                        cx, DERIVE_HASH_NOT_EQ, span,
                        "you are deriving `Hash` but have implemented `PartialEq` explicitly",
                        |db| {
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

/// Implementation of the `EXPL_IMPL_CLONE_ON_COPY` lint.
fn check_copy_clone<'a, 'tcx>(cx: &LateContext<'a, 'tcx>,
                              item: &Item,
                              trait_ref: &TraitRef, ty: ty::Ty<'tcx>) {
    if match_path(&trait_ref.path, &CLONE_TRAIT_PATH) {
        let parameter_environment = ty::ParameterEnvironment::for_item(cx.tcx, item.id);
        let subst_ty = ty.subst(cx.tcx, &parameter_environment.free_substs);

        if subst_ty.moves_by_default(&parameter_environment, item.span) {
            return; // ty is not Copy
        }

        // Some types are not Clone by default but could be cloned `by hand` if necessary
        match ty.sty {
            TypeVariants::TyEnum(def, substs) | TypeVariants::TyStruct(def, substs) => {
                for variant in &def.variants {
                    for field in &variant.fields {
                        match field.ty(cx.tcx, substs).sty {
                            TypeVariants::TyArray(_, size) if size > 32 => {
                                return;
                            }
                            TypeVariants::TyBareFn(..) => {
                                return;
                            }
                            TypeVariants::TyTuple(ref tys) if tys.len() > 12 => {
                                return;
                            }
                            _ => (),
                        }
                    }
                }
            }
            _ => (),
        }

        span_lint_and_then(cx,
                           DERIVE_HASH_NOT_EQ,
                           item.span,
                           "you are implementing `Clone` explicitly on a `Copy` type",
                           |db| {
                               db.span_note(item.span, "consider deriving `Clone` or removing `Copy`");
                           });
    }
}

/// Checks for the `#[automatically_derived]` attribute all `#[derive]`d implementations have.
fn is_automatically_derived(attr: &Attribute) -> bool {
    if let MetaItemKind::Word(ref word) = attr.node.value.node {
        word == &"automatically_derived"
    } else {
        false
    }
}
