use crate::utils::{in_macro, span_lint_and_then};
use rustc::hir::intravisit::{walk_path, walk_ty, NestedVisitorMap, Visitor};
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::ty;
use syntax::ast::NodeId;
use syntax::symbol::keywords;
use syntax_pos::symbol::keywords::SelfType;

/// **What it does:** Checks for unnecessary repetition of structure name when a
/// replacement with `Self` is applicable.
///
/// **Why is this bad?** Unnecessary repetition. Mixed use of `Self` and struct
/// name
/// feels inconsistent.
///
/// **Known problems:** None.
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
/// ```
/// struct Foo {}
/// impl Foo {
///     fn new() -> Self {
///         Self {}
///     }
/// }
/// ```
declare_clippy_lint! {
    pub USE_SELF,
    pedantic,
    "Unnecessary structure name repetition whereas `Self` is applicable"
}

#[derive(Copy, Clone, Default)]
pub struct UseSelf;

impl LintPass for UseSelf {
    fn get_lints(&self) -> LintArray {
        lint_array!(USE_SELF)
    }
}

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

fn span_use_self_lint(cx: &LateContext, path: &Path) {
    span_lint_and_then(cx, USE_SELF, path.span, "unnecessary structure name repetition", |db| {
        db.span_suggestion(path.span, "use the applicable keyword", "Self".to_owned());
    });
}

struct TraitImplTyVisitor<'a, 'tcx: 'a> {
    cx: &'a LateContext<'a, 'tcx>,
    type_walker: ty::walk::TypeWalker<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for TraitImplTyVisitor<'a, 'tcx> {
    fn visit_ty(&mut self, t: &'tcx Ty) {
        let trait_ty = self.type_walker.next();
        if let TyKind::Path(QPath::Resolved(_, path)) = &t.node {
            let impl_is_self_ty = if let def::Def::SelfTy(..) = path.def {
                true
            } else {
                false
            };
            if !impl_is_self_ty {
                let trait_is_self_ty = if let Some(ty::TyParam(ty::ParamTy { name, .. })) = trait_ty.map(|ty| &ty.sty) {
                    *name == keywords::SelfType.name().as_str()
                } else {
                    false
                };
                if trait_is_self_ty {
                    span_use_self_lint(self.cx, path);
                }
            }
        }
        walk_ty(self, t)
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::None
    }
}

fn check_trait_method_impl_decl<'a, 'tcx: 'a>(
    cx: &'a LateContext<'a, 'tcx>,
    impl_item: &ImplItem,
    impl_decl: &'tcx FnDecl,
    impl_trait_ref: &ty::TraitRef,
) {
    let trait_method = cx
        .tcx
        .associated_items(impl_trait_ref.def_id)
        .find(|assoc_item| {
            assoc_item.kind == ty::AssociatedKind::Method
                && cx
                    .tcx
                    .hygienic_eq(impl_item.ident, assoc_item.ident, impl_trait_ref.def_id)
        })
        .expect("impl method matches a trait method");

    let trait_method_sig = cx.tcx.fn_sig(trait_method.def_id);
    let trait_method_sig = cx.tcx.erase_late_bound_regions(&trait_method_sig);

    let output_ty = if let FunctionRetTy::Return(ty) = &impl_decl.output {
        Some(&**ty)
    } else {
        None
    };

    for (impl_ty, trait_ty) in impl_decl
        .inputs
        .iter()
        .chain(output_ty)
        .zip(trait_method_sig.inputs_and_output)
    {
        let mut visitor = TraitImplTyVisitor {
            cx,
            type_walker: trait_ty.walk(),
        };

        visitor.visit_ty(&impl_ty);
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'a, 'tcx>, item: &'tcx Item) {
        if in_macro(item.span) {
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
                        GenericArg::Type(_) => false,
                    })
                } else {
                    true
                };

                if should_check {
                    let visitor = &mut UseSelfVisitor {
                        item_path,
                        cx,
                    };
                    let impl_def_id = cx.tcx.hir.local_def_id(item.id);
                    let impl_trait_ref = cx.tcx.impl_trait_ref(impl_def_id);

                    if let Some(impl_trait_ref) = impl_trait_ref {
                        for impl_item_ref in refs {
                            let impl_item = cx.tcx.hir.impl_item(impl_item_ref.id);
                            if let ImplItemKind::Method(MethodSig{ decl: impl_decl, .. }, impl_body_id)
                                    = &impl_item.node {
                                check_trait_method_impl_decl(cx, impl_item, impl_decl, &impl_trait_ref);
                                let body = cx.tcx.hir.body(*impl_body_id);
                                visitor.visit_body(body);
                            } else {
                                visitor.visit_impl_item(impl_item);
                            }
                        }
                    } else {
                        for impl_item_ref in refs {
                            let impl_item = cx.tcx.hir.impl_item(impl_item_ref.id);
                            visitor.visit_impl_item(impl_item);
                        }
                    }
                }
            }
        }
    }
}

struct UseSelfVisitor<'a, 'tcx: 'a> {
    item_path: &'a Path,
    cx: &'a LateContext<'a, 'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for UseSelfVisitor<'a, 'tcx> {
    fn visit_path(&mut self, path: &'tcx Path, _id: NodeId) {
        if self.item_path.def == path.def && path.segments.last().expect(SEGMENTS_MSG).ident.name != SelfType.name() {
            span_use_self_lint(self.cx, path);
        }

        walk_path(self, path);
    }

    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'tcx> {
        NestedVisitorMap::OnlyBodies(&self.cx.tcx.hir)
    }
}
