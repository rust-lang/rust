use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::intravisit::{walk_item, walk_path, walk_ty, NestedVisitorMap, Visitor};
use rustc_hir::{
    def, FnDecl, FnRetTy, FnSig, GenericArg, HirId, ImplItem, ImplItemKind, Item, ItemKind, Path, PathSegment, QPath,
    TyKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_middle::ty::{DefIdTree, Ty};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::kw;
use rustc_typeck::hir_ty_to_ty;

use crate::utils::{differing_macro_contexts, meets_msrv, span_lint_and_sugg};

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
    "unnecessary structure name repetition whereas `Self` is applicable"
}

impl_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

fn span_use_self_lint(cx: &LateContext<'_>, path: &Path<'_>, last_segment: Option<&PathSegment<'_>>) {
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

// FIXME: always use this (more correct) visitor, not just in method signatures.
struct SemanticUseSelfVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    self_ty: Ty<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for SemanticUseSelfVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'_>) {
        if let TyKind::Path(QPath::Resolved(_, path)) = &hir_ty.kind {
            match path.res {
                def::Res::SelfTy(..) => {},
                _ => {
                    if hir_ty_to_ty(self.cx.tcx, hir_ty) == self.self_ty {
                        span_use_self_lint(self.cx, path, None);
                    }
                },
            }
        }

        walk_ty(self, hir_ty)
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

fn check_trait_method_impl_decl<'tcx>(
    cx: &LateContext<'tcx>,
    impl_item: &ImplItem<'_>,
    impl_decl: &'tcx FnDecl<'_>,
    impl_trait_ref: ty::TraitRef<'tcx>,
) {
    let trait_method = cx
        .tcx
        .associated_items(impl_trait_ref.def_id)
        .find_by_name_and_kind(cx.tcx, impl_item.ident, ty::AssocKind::Fn, impl_trait_ref.def_id)
        .expect("impl method matches a trait method");

    let trait_method_sig = cx.tcx.fn_sig(trait_method.def_id);
    let trait_method_sig = cx.tcx.erase_late_bound_regions(trait_method_sig);

    let output_hir_ty = if let FnRetTy::Return(ty) = &impl_decl.output {
        Some(&**ty)
    } else {
        None
    };

    // `impl_hir_ty` (of type `hir::Ty`) represents the type written in the signature.
    // `trait_ty` (of type `ty::Ty`) is the semantic type for the signature in the trait.
    // We use `impl_hir_ty` to see if the type was written as `Self`,
    // `hir_ty_to_ty(...)` to check semantic types of paths, and
    // `trait_ty` to determine which parts of the signature in the trait, mention
    // the type being implemented verbatim (as opposed to `Self`).
    for (impl_hir_ty, trait_ty) in impl_decl
        .inputs
        .iter()
        .chain(output_hir_ty)
        .zip(trait_method_sig.inputs_and_output)
    {
        // Check if the input/output type in the trait method specifies the implemented
        // type verbatim, and only suggest `Self` if that isn't the case.
        // This avoids suggestions to e.g. replace `Vec<u8>` with `Vec<Self>`,
        // in an `impl Trait for u8`, when the trait always uses `Vec<u8>`.
        // See also https://github.com/rust-lang/rust-clippy/issues/2894.
        let self_ty = impl_trait_ref.self_ty();
        if !trait_ty.walk().any(|inner| inner == self_ty.into()) {
            let mut visitor = SemanticUseSelfVisitor { cx, self_ty };

            visitor.visit_ty(&impl_hir_ty);
        }
    }
}

const USE_SELF_MSRV: RustcVersion = RustcVersion::new(1, 37, 0);

pub struct UseSelf {
    msrv: Option<RustcVersion>,
}

impl UseSelf {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self { msrv }
    }
}

impl<'tcx> LateLintPass<'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if !meets_msrv(self.msrv.as_ref(), &USE_SELF_MSRV) {
            return;
        }

        if in_external_macro(cx.sess(), item.span) {
            return;
        }
        if_chain! {
            if let ItemKind::Impl{ self_ty: ref item_type, items: refs, .. } = item.kind;
            if let TyKind::Path(QPath::Resolved(_, ref item_path)) = item_type.kind;
            then {
                let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args;
                let should_check = parameters.as_ref().map_or(
                    true,
                    |params| !params.parenthesized
                        &&!params.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
                );

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
                            if let ImplItemKind::Fn(FnSig{ decl: impl_decl, .. }, impl_body_id)
                                    = &impl_item.kind {
                                check_trait_method_impl_decl(cx, impl_item, impl_decl, impl_trait_ref);

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
    extract_msrv_attr!(LateContext);
}

struct UseSelfVisitor<'a, 'tcx> {
    item_path: &'a Path<'a>,
    cx: &'a LateContext<'tcx>,
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

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::All(self.cx.tcx.hir())
    }
}
