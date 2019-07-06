use if_chain::if_chain;
use rustc::hir;
use rustc::hir::def::{CtorKind, DefKind, Res};
use rustc::hir::intravisit::{walk_item, walk_path, walk_ty, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{in_external_macro, LateContext, LateLintPass, LintArray, LintContext, LintPass};
use rustc::ty;
use rustc::ty::{DefIdTree, Ty};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax_pos::symbol::kw;

use crate::utils::span_lint_and_sugg;

declare_clippy_lint! {
    /// **What it does:** Checks for unnecessary repetition of structure name when a
    /// replacement with `Self` is applicable.
    ///
    /// **Why is this bad?** Unnecessary repetition. Mixed use of `Self` and struct
    /// name
    /// feels inconsistent.
    ///
    /// **Known problems:**
    /// - False positive when using associated types (#2843)
    /// - False positives in some situations when using generics (#3410)
    ///
    /// **Example:**
    /// ```rust
    /// struct Foo {}
    /// impl Foo {
    ///     fn new() -> Foo {
    ///         Foo {}
    ///     }
    /// }
    /// ```
    /// could be
    /// ```rust
    /// struct Foo {}
    /// impl Foo {
    ///     fn new() -> Self {
    ///         Self {}
    ///     }
    /// }
    /// ```
    pub USE_SELF,
    pedantic,
    "Unnecessary structure name repetition whereas `Self` is applicable"
}

declare_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

fn span_use_self_lint(cx: &LateContext<'_, '_>, path: &Path) {
    // Path segments only include actual path, no methods or fields.
    let last_path_span = path.segments.last().expect(SEGMENTS_MSG).ident.span;
    // Only take path up to the end of last_path_span.
    let span = path.span.with_hi(last_path_span.hi());

    span_lint_and_sugg(
        cx,
        USE_SELF,
        span,
        "unnecessary structure name repetition",
        "use the applicable keyword",
        "Self".to_owned(),
        Applicability::MachineApplicable,
    );
}

struct TraitImplTyVisitor<'a, 'tcx> {
    item_type: Ty<'tcx>,
    cx: &'a LateContext<'a, 'tcx>,
    trait_type_walker: ty::walk::TypeWalker<'tcx>,
    impl_type_walker: ty::walk::TypeWalker<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for TraitImplTyVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, t: &'tcx hir::Ty) {
        let trait_ty = self.trait_type_walker.next();
        let impl_ty = self.impl_type_walker.next();

        if let TyKind::Path(QPath::Resolved(_, path)) = &t.node {
            // The implementation and trait types don't match which means that
            // the concrete type was specified by the implementation
            if impl_ty != trait_ty {
                if let Some(impl_ty) = impl_ty {
                    if self.item_type == impl_ty {
                        let is_self_ty = if let def::Res::SelfTy(..) = path.res {
                            true
                        } else {
                            false
                        };

                        if !is_self_ty {
                            span_use_self_lint(self.cx, path);
                        }
                    }
                }
            }
        }

        walk_ty(self, t)
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn check_trait_method_impl_decl<'a, 'tcx>(
    cx: &'a LateContext<'a, 'tcx>,
    item_type: Ty<'tcx>,
    impl_item: &ImplItem,
    impl_decl: &'tcx FnDecl,
    impl_trait_ref: &ty::TraitRef<'_>,
) {
    let trait_method = cx
        .tcx
        .associated_items(impl_trait_ref.def_id)
        .find(|assoc_item| {
            assoc_item.kind == ty::AssocKind::Method
                && cx
                    .tcx
                    .hygienic_eq(impl_item.ident, assoc_item.ident, impl_trait_ref.def_id)
        })
        .expect("impl method matches a trait method");

    let trait_method_sig = cx.tcx.fn_sig(trait_method.def_id);
    let trait_method_sig = cx.tcx.erase_late_bound_regions(&trait_method_sig);

    let impl_method_def_id = cx.tcx.hir().local_def_id(impl_item.hir_id);
    let impl_method_sig = cx.tcx.fn_sig(impl_method_def_id);
    let impl_method_sig = cx.tcx.erase_late_bound_regions(&impl_method_sig);

    let output_ty = if let FunctionRetTy::Return(ty) = &impl_decl.output {
        Some(&**ty)
    } else {
        None
    };

    // `impl_decl_ty` (of type `hir::Ty`) represents the type declared in the signature.
    // `impl_ty` (of type `ty:TyS`) is the concrete type that the compiler has determined for
    // that declaration. We use `impl_decl_ty` to see if the type was declared as `Self`
    // and use `impl_ty` to check its concrete type.
    for (impl_decl_ty, (impl_ty, trait_ty)) in impl_decl.inputs.iter().chain(output_ty).zip(
        impl_method_sig
            .inputs_and_output
            .iter()
            .zip(trait_method_sig.inputs_and_output),
    ) {
        let mut visitor = TraitImplTyVisitor {
            cx,
            item_type,
            trait_type_walker: trait_ty.walk(),
            impl_type_walker: impl_ty.walk(),
        };

        visitor.visit_ty(&impl_decl_ty);
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if in_external_macro(cx.sess(), item.span) {
            return;
        }
        if_chain! {
            if let ItemKind::Impl(.., ref item_type, ref refs) = item.node;
            if let TyKind::Path(QPath::Resolved(_, ref item_path)) = item_type.node;
            then {
                let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args;
                let should_check = if let Some(ref params) = *parameters {
                    !params.parenthesized && !params.args.iter().any(|arg| match arg {
                        GenericArg::Lifetime(_) => true,
                        _ => false,
                    })
                } else {
                    true
                };

                if should_check {
                    let visitor = &mut UseSelfVisitor {
                        item_path,
                        cx,
                    };
                    let impl_def_id = cx.tcx.hir().local_def_id(item.hir_id);
                    let impl_trait_ref = cx.tcx.impl_trait_ref(impl_def_id);

                    if let Some(impl_trait_ref) = impl_trait_ref {
                        for impl_item_ref in refs {
                            let impl_item = cx.tcx.hir().impl_item(impl_item_ref.id);
                            if let ImplItemKind::Method(MethodSig{ decl: impl_decl, .. }, impl_body_id)
                                    = &impl_item.node {
                                let item_type = cx.tcx.type_of(impl_def_id);
                                check_trait_method_impl_decl(cx, item_type, impl_item, impl_decl, &impl_trait_ref);

                                let body = cx.tcx.hir().body(*impl_body_id);
                                visitor.visit_body(body);
                            } else {
                                visitor.visit_impl_item(impl_item);
                            }
                        }
                    } else {
                        for impl_item_ref in refs {
                            let impl_item = cx.tcx.hir().impl_item(impl_item_ref.id);
                            visitor.visit_impl_item(impl_item);
                        }
                    }
                }
            }
        }
    }
}

struct UseSelfVisitor<'a, 'tcx> {
    item_path: &'a Path,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for UseSelfVisitor<'a, 'tcx> {
    fn visit_path(&mut self, path: &'tcx Path, _id: HirId) {
        if path.segments.last().expect(SEGMENTS_MSG).ident.name != kw::SelfUpper {
            if self.item_path.res == path.res {
                span_use_self_lint(self.cx, path);
            } else if let Res::Def(DefKind::Ctor(def::CtorOf::Struct, CtorKind::Fn), ctor_did) = path.res {
                if self.item_path.res.opt_def_id() == self.cx.tcx.parent(ctor_did) {
                    span_use_self_lint(self.cx, path);
                }
            }
        }
        walk_path(self, path);
    }

    fn visit_item(&mut self, item: &'tcx Item) {
        match item.node {
            ItemKind::Use(..)
            | ItemKind::Static(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..)
            | ItemKind::Impl(..)
            | ItemKind::Fn(..) => {
                // Don't check statements that shadow `Self` or where `Self` can't be used
            },
            _ => walk_item(self, item),
        }
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}
