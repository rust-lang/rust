use crate::utils;
use crate::utils::snippet_opt;
use crate::utils::span_lint_and_sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::intravisit::{walk_expr, walk_impl_item, walk_ty, NestedVisitorMap, Visitor};
use rustc_hir::{
    def, Expr, ExprKind, FnDecl, FnRetTy, FnSig, GenericArg, ImplItem, ImplItemKind, ItemKind, Node, Path, PathSegment,
    QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty;
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::{BytePos, Span};
use rustc_typeck::hir_ty_to_ty;

declare_clippy_lint! {
    /// **What it does:** Checks for unnecessary repetition of structure name when a
    /// replacement with `Self` is applicable.
    ///
    /// **Why is this bad?** Unnecessary repetition. Mixed use of `Self` and struct
    /// name
    /// feels inconsistent.
    ///
    /// **Known problems:**
    /// Unaddressed false negatives related to unresolved internal compiler errors.
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

declare_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

fn span_lint(cx: &LateContext<'_>, span: Span) {
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

#[allow(clippy::cast_possible_truncation)]
fn span_lint_until_last_segment<'tcx>(cx: &LateContext<'tcx>, span: Span, segment: &'tcx PathSegment<'tcx>) {
    let sp = span.with_hi(segment.ident.span.lo());
    // remove the trailing ::
    let span_without_last_segment = match snippet_opt(cx, sp) {
        Some(snippet) => match snippet.rfind("::") {
            Some(bidx) => sp.with_hi(sp.lo() + BytePos(bidx as u32)),
            None => sp,
        },
        None => sp,
    };
    span_lint(cx, span_without_last_segment);
}

fn span_lint_on_path_until_last_segment<'tcx>(cx: &LateContext<'tcx>, path: &'tcx Path<'tcx>) {
    if path.segments.len() > 1 {
        span_lint_until_last_segment(cx, path.span, path.segments.last().unwrap());
    }
}

fn span_lint_on_qpath_resolved<'tcx>(cx: &LateContext<'tcx>, qpath: &'tcx QPath<'tcx>, until_last_segment: bool) {
    if let QPath::Resolved(_, path) = qpath {
        if until_last_segment {
            span_lint_on_path_until_last_segment(cx, path);
        } else {
            span_lint(cx, path.span);
        }
    }
}

struct BodyVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    self_ty: Ty<'tcx>,
}

impl<'a, 'tcx> BodyVisitor<'a, 'tcx> {
    fn check_trait_method_impl_decl(
        &mut self,
        impl_item: &ImplItem<'tcx>,
        impl_decl: &'tcx FnDecl<'tcx>,
        impl_trait_ref: ty::TraitRef<'tcx>,
    ) {
        let tcx = self.cx.tcx;
        let trait_method = tcx
            .associated_items(impl_trait_ref.def_id)
            .find_by_name_and_kind(tcx, impl_item.ident, ty::AssocKind::Fn, impl_trait_ref.def_id)
            .expect("impl method matches a trait method");

        let trait_method_sig = tcx.fn_sig(trait_method.def_id);
        let trait_method_sig = tcx.erase_late_bound_regions(&trait_method_sig);

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
                self.visit_ty(&impl_hir_ty);
            }
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for BodyVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::OnlyBodies(self.cx.tcx.hir())
    }

    fn visit_expr(&mut self, expr: &'tcx Expr<'tcx>) {
        fn expr_ty_matches<'tcx>(expr: &'tcx Expr<'tcx>, self_ty: Ty<'tcx>, cx: &LateContext<'tcx>) -> bool {
            let def_id = expr.hir_id.owner;
            if cx.tcx.has_typeck_results(def_id) {
                cx.tcx.typeck(def_id).expr_ty_opt(expr) == Some(self_ty)
            } else {
                false
            }
        }
        match expr.kind {
            ExprKind::Struct(QPath::Resolved(_, path), ..) => {
                if expr_ty_matches(expr, self.self_ty, self.cx) {
                    match path.res {
                        def::Res::SelfTy(..) => (),
                        def::Res::Def(DefKind::Variant, _) => span_lint_on_path_until_last_segment(self.cx, path),
                        _ => {
                            span_lint(self.cx, path.span);
                        },
                    }
                }
            },
            // tuple struct instantiation (`Foo(arg)` or `Enum::Foo(arg)`)
            ExprKind::Call(fun, _) => {
                if let Expr {
                    kind: ExprKind::Path(ref qpath),
                    ..
                } = fun
                {
                    if expr_ty_matches(expr, self.self_ty, self.cx) {
                        let res = utils::qpath_res(self.cx, qpath, fun.hir_id);

                        if let def::Res::Def(DefKind::Ctor(ctor_of, _), ..) = res {
                            match ctor_of {
                                def::CtorOf::Variant => {
                                    span_lint_on_qpath_resolved(self.cx, qpath, true);
                                },
                                def::CtorOf::Struct => {
                                    span_lint_on_qpath_resolved(self.cx, qpath, false);
                                },
                            }
                        }
                    }
                }
            },
            // unit enum variants (`Enum::A`)
            ExprKind::Path(ref qpath) => {
                if expr_ty_matches(expr, self.self_ty, self.cx) {
                    span_lint_on_qpath_resolved(self.cx, qpath, true);
                }
            },
            _ => (),
        }
        walk_expr(self, expr);
    }
}

struct FnSigVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    self_ty: Ty<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for FnSigVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'tcx>) {
        if let TyKind::Path(QPath::Resolved(_, path)) = hir_ty.kind {
            match path.res {
                def::Res::SelfTy(..) => {},
                _ => {
                    match self.cx.tcx.hir().find(self.cx.tcx.hir().get_parent_node(hir_ty.hir_id)) {
                        Some(Node::Expr(Expr {
                            kind: ExprKind::Path(QPath::TypeRelative(_, segment)),
                            ..
                        })) => {
                            // The following block correctly identifies applicable lint locations
                            // but `hir_ty_to_ty` calls cause odd ICEs.
                            //
                            if hir_ty_to_ty(self.cx.tcx, hir_ty) == self.self_ty {
                                // fixme: this span manipulation should not be necessary
                                // @flip1995 found an ast lowering issue in
                                // https://github.com/rust-lang/rust/blob/master/src/librustc_ast_lowering/path.rs#l142-l162
                                span_lint_until_last_segment(self.cx, hir_ty.span, segment);
                            }
                        },
                        _ => {
                            if hir_ty_to_ty(self.cx.tcx, hir_ty) == self.self_ty {
                                span_lint(self.cx, hir_ty.span)
                            }
                        },
                    }
                },
            }
        }

        walk_ty(self, hir_ty);
    }
}

impl<'tcx> LateLintPass<'tcx> for UseSelf {
    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, impl_item: &'tcx ImplItem<'_>) {
        if in_external_macro(cx.sess(), impl_item.span) {
            return;
        }

        let parent_id = cx.tcx.hir().get_parent_item(impl_item.hir_id);
        let imp = cx.tcx.hir().expect_item(parent_id);

        if_chain! {
            if let ItemKind::Impl { self_ty: hir_self_ty, .. } = imp.kind;
            if let TyKind::Path(QPath::Resolved(_, ref item_path)) = hir_self_ty.kind;
            then {
                let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args;
                let should_check = parameters.as_ref().map_or(
                    true,
                    |params| !params.parenthesized
                        &&!params.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
                );

                // TODO: don't short-circuit upon lifetime parameters
                if should_check {
                    let self_ty = hir_ty_to_ty(cx.tcx, hir_self_ty);
                    let body_visitor = &mut BodyVisitor { cx, self_ty };
                    let fn_sig_visitor = &mut FnSigVisitor { cx, self_ty };

                    let tcx = cx.tcx;
                    let impl_def_id = tcx.hir().local_def_id(imp.hir_id);
                    let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);
                    if_chain! {
                        if let Some(impl_trait_ref) = impl_trait_ref;
                        if let ImplItemKind::Fn(FnSig { decl: impl_decl, .. }, impl_body_id) = &impl_item.kind;
                        then {
                            body_visitor.check_trait_method_impl_decl(impl_item, impl_decl, impl_trait_ref);
                            let body = tcx.hir().body(*impl_body_id);
                            body_visitor.visit_body(body);
                        } else {
                            walk_impl_item(body_visitor, impl_item);
                            walk_impl_item(fn_sig_visitor, impl_item);
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
