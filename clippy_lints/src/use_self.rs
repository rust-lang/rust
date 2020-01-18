use if_chain::if_chain;
use rustc::hir::map::Map;
use rustc::lint::in_external_macro;
use rustc::ty;
use rustc::ty::{DefIdTree, Ty};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_item, walk_path, walk_ty, NestedVisitorMap, Visitor};
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::kw;

use crate::utils::{differing_macro_contexts, span_lint_and_sugg};

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
    nursery,
    "Unnecessary structure name repetition whereas `Self` is applicable"
}

declare_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

fn span_use_self_lint(cx: &LateContext<'_, '_>, path: &Path<'_>, last_segment: Option<&PathSegment<'_>>) {
    let last_segment = last_segment.unwrap_or_else(|| path.segments.last().expect(SEGMENTS_MSG));

    // Path segments only include actual path, no methods or fields.
    let last_path_span = last_segment.ident.span;

    if differing_macro_contexts(path.span, last_path_span) {
        return;
    }

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
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, t: &'tcx hir::Ty<'_>) {
        let trait_ty = self.trait_type_walker.next();
        let impl_ty = self.impl_type_walker.next();

        if_chain! {
            if let TyKind::Path(QPath::Resolved(_, path)) = &t.kind;

            // The implementation and trait types don't match which means that
            // the concrete type was specified by the implementation
            if impl_ty != trait_ty;
            if let Some(impl_ty) = impl_ty;
            if self.item_type == impl_ty;
            then {
                match path.res {
                    def::Res::SelfTy(..) => {},
                    _ => span_use_self_lint(self.cx, path, None)
                }
            }
        }

        walk_ty(self, t)
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::None
    }
}

fn check_trait_method_impl_decl<'a, 'tcx>(
    cx: &'a LateContext<'a, 'tcx>,
    item_type: Ty<'tcx>,
    impl_item: &ImplItem<'_>,
    impl_decl: &'tcx FnDecl<'_>,
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
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item<'_>) {
        if in_external_macro(cx.sess(), item.span) {
            return;
        }
        if_chain! {
            if let ItemKind::Impl{ self_ty: ref item_type, items: refs, .. } = item.kind;
            if let TyKind::Path(QPath::Resolved(_, ref item_path)) = item_type.kind;
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
                            if let ImplItemKind::Method(FnSig{ decl: impl_decl, .. }, impl_body_id)
                                    = &impl_item.kind {
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
    item_path: &'a Path<'a>,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for UseSelfVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_path(&mut self, path: &'tcx Path<'_>, _id: HirId) {
        if !path.segments.iter().any(|p| p.ident.span.is_dummy()) {
            if path.segments.len() >= 2 {
                let last_but_one = &path.segments[path.segments.len() - 2];
                if last_but_one.ident.name != kw::SelfUpper {
                    let enum_def_id = match path.res {
                        Res::Def(DefKind::Variant, variant_def_id) => self.cx.tcx.parent(variant_def_id),
                        Res::Def(DefKind::Ctor(def::CtorOf::Variant, _), ctor_def_id) => {
                            let variant_def_id = self.cx.tcx.parent(ctor_def_id);
                            variant_def_id.and_then(|def_id| self.cx.tcx.parent(def_id))
                        },
                        _ => None,
                    };

                    if self.item_path.res.opt_def_id() == enum_def_id {
                        span_use_self_lint(self.cx, path, Some(last_but_one));
                    }
                }
            }

            if path.segments.last().expect(SEGMENTS_MSG).ident.name != kw::SelfUpper {
                if self.item_path.res == path.res {
                    span_use_self_lint(self.cx, path, None);
                } else if let Res::Def(DefKind::Ctor(def::CtorOf::Struct, _), ctor_def_id) = path.res {
                    if self.item_path.res.opt_def_id() == self.cx.tcx.parent(ctor_def_id) {
                        span_use_self_lint(self.cx, path, None);
                    }
                }
            }
        }

        walk_path(self, path);
    }

    fn visit_item(&mut self, item: &'tcx Item<'_>) {
        match item.kind {
            ItemKind::Use(..)
            | ItemKind::Static(..)
            | ItemKind::Enum(..)
            | ItemKind::Struct(..)
            | ItemKind::Union(..)
            | ItemKind::Impl { .. }
            | ItemKind::Fn(..) => {
                // Don't check statements that shadow `Self` or where `Self` can't be used
            },
            _ => walk_item(self, item),
        }
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<'_, Self::Map> {
        NestedVisitorMap::All(&self.cx.tcx.hir())
    }
}
