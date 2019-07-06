use crate::utils::paths;
use crate::utils::{is_automatically_derived, is_copy, match_path, span_lint_and_then};
use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty::{self, Ty};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::source_map::Span;

declare_clippy_lint! {
    /// **What it does:** Checks for deriving `Hash` but implementing `PartialEq`
    /// explicitly or vice versa.
    ///
    /// **Why is this bad?** The implementation of these traits must agree (for
    /// example for use with `HashMap`) so it’s probably a bad idea to use a
    /// default-generated `Hash` implementation with an explicitly defined
    /// `PartialEq`. In particular, the following must hold for any type:
    ///
    /// ```text
    /// k1 == k2 ⇒ hash(k1) == hash(k2)
    /// ```
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```ignore
    /// #[derive(Hash)]
    /// struct Foo;
    ///
    /// impl PartialEq for Foo {
    ///     ...
    /// }
    /// ```
    pub DERIVE_HASH_XOR_EQ,
    correctness,
    "deriving `Hash` but implementing `PartialEq` explicitly"
}

declare_clippy_lint! {
    /// **What it does:** Checks for explicit `Clone` implementations for `Copy`
    /// types.
    ///
    /// **Why is this bad?** To avoid surprising behaviour, these traits should
    /// agree and the behaviour of `Copy` cannot be overridden. In almost all
    /// situations a `Copy` type should have a `Clone` implementation that does
    /// nothing more than copy the object, which is what `#[derive(Copy, Clone)]`
    /// gets you.
    ///
    /// **Known problems:** Bounds of generic types are sometimes wrong: https://github.com/rust-lang/rust/issues/26925
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
    pub EXPL_IMPL_CLONE_ON_COPY,
    pedantic,
    "implementing `Clone` explicitly on `Copy` types"
}

declare_lint_pass!(Derive => [EXPL_IMPL_CLONE_ON_COPY, DERIVE_HASH_XOR_EQ]);

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for Derive {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if let ItemKind::Impl(_, _, _, _, Some(ref trait_ref), _, _) = item.node {
            let ty = cx.tcx.type_of(cx.tcx.hir().local_def_id(item.hir_id));
            let is_automatically_derived = is_automatically_derived(&*item.attrs);

            check_hash_peq(cx, item.span, trait_ref, ty, is_automatically_derived);

            if !is_automatically_derived {
                check_copy_clone(cx, item, trait_ref, ty);
            }
        }
    }
}

/// Implementation of the `DERIVE_HASH_XOR_EQ` lint.
fn check_hash_peq<'a, 'tcx>(
    cx: &LateContext<'a, 'tcx>,
    span: Span,
    trait_ref: &TraitRef,
    ty: Ty<'tcx>,
    hash_is_automatically_derived: bool,
) {
    if_chain! {
        if match_path(&trait_ref.path, &paths::HASH);
        if let Some(peq_trait_def_id) = cx.tcx.lang_items().eq_trait();
        then {
            // Look for the PartialEq implementations for `ty`
            cx.tcx.for_each_relevant_impl(peq_trait_def_id, ty, |impl_id| {
                let peq_is_automatically_derived = is_automatically_derived(&cx.tcx.get_attrs(impl_id));

                if peq_is_automatically_derived == hash_is_automatically_derived {
                    return;
                }

                let trait_ref = cx.tcx.impl_trait_ref(impl_id).expect("must be a trait implementation");

                // Only care about `impl PartialEq<Foo> for Foo`
                // For `impl PartialEq<B> for A, input_types is [A, B]
                if trait_ref.substs.type_at(1) == ty {
                    let mess = if peq_is_automatically_derived {
                        "you are implementing `Hash` explicitly but have derived `PartialEq`"
                    } else {
                        "you are deriving `Hash` but have implemented `PartialEq` explicitly"
                    };

                    span_lint_and_then(
                        cx, DERIVE_HASH_XOR_EQ, span,
                        mess,
                        |db| {
                        if let Some(node_id) = cx.tcx.hir().as_local_hir_id(impl_id) {
                            db.span_note(
                                cx.tcx.hir().span(node_id),
                                "`PartialEq` implemented here"
                            );
                        }
                    });
                }
            });
        }
    }
}

/// Implementation of the `EXPL_IMPL_CLONE_ON_COPY` lint.
fn check_copy_clone<'a, 'tcx>(cx: &LateContext<'a, 'tcx>, item: &Item, trait_ref: &TraitRef, ty: Ty<'tcx>) {
    if match_path(&trait_ref.path, &paths::CLONE_TRAIT) {
        if !is_copy(cx, ty) {
            return;
        }

        match ty.sty {
            ty::Adt(def, _) if def.is_union() => return,

            // Some types are not Clone by default but could be cloned “by hand” if necessary
            ty::Adt(def, substs) => {
                for variant in &def.variants {
                    for field in &variant.fields {
                        if let ty::FnDef(..) = field.ty(cx.tcx, substs).sty {
                            return;
                        }
                    }
                    for subst in substs {
                        if let ty::subst::UnpackedKind::Type(subst) = subst.unpack() {
                            if let ty::Param(_) = subst.sty {
                                return;
                            }
                        }
                    }
                }
            },
            _ => (),
        }

        span_lint_and_then(
            cx,
            EXPL_IMPL_CLONE_ON_COPY,
            item.span,
            "you are implementing `Clone` explicitly on a `Copy` type",
            |db| {
                db.span_note(item.span, "consider deriving `Clone` or removing `Copy`");
            },
        );
    }
}
