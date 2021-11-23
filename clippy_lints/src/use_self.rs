use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::same_type_and_consts;
use clippy_utils::{meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir,
    def::{CtorOf, DefKind, Res},
    def_id::LocalDefId,
    intravisit::{walk_inf, walk_ty, NestedVisitorMap, Visitor},
    Expr, ExprKind, FnRetTy, FnSig, GenericArg, HirId, Impl, ImplItemKind, Item, ItemKind, Path, QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::AssocKind;
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
use rustc_typeck::hir_ty_to_ty;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for unnecessary repetition of structure name when a
    /// replacement with `Self` is applicable.
    ///
    /// ### Why is this bad?
    /// Unnecessary repetition. Mixed use of `Self` and struct
    /// name
    /// feels inconsistent.
    ///
    /// ### Known problems
    /// - Unaddressed false negative in fn bodies of trait implementations
    /// - False positive with assotiated types in traits (#4140)
    ///
    /// ### Example
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
    #[clippy::version = "pre 1.29.0"]
    pub USE_SELF,
    nursery,
    "unnecessary structure name repetition whereas `Self` is applicable"
}

#[derive(Default)]
pub struct UseSelf {
    msrv: Option<RustcVersion>,
    stack: Vec<StackItem>,
}

impl UseSelf {
    #[must_use]
    pub fn new(msrv: Option<RustcVersion>) -> Self {
        Self {
            msrv,
            ..Self::default()
        }
    }
}

#[derive(Debug)]
enum StackItem {
    Check {
        impl_id: LocalDefId,
        in_body: u32,
        types_to_skip: FxHashSet<HirId>,
    },
    NoCheck,
}

impl_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

impl<'tcx> LateLintPass<'tcx> for UseSelf {
    fn check_item(&mut self, _cx: &LateContext<'_>, item: &Item<'_>) {
        if matches!(item.kind, ItemKind::OpaqueTy(_)) {
            // skip over `ItemKind::OpaqueTy` in order to lint `foo() -> impl <..>`
            return;
        }
        // We push the self types of `impl`s on a stack here. Only the top type on the stack is
        // relevant for linting, since this is the self type of the `impl` we're currently in. To
        // avoid linting on nested items, we push `StackItem::NoCheck` on the stack to signal, that
        // we're in an `impl` or nested item, that we don't want to lint
        let stack_item = if_chain! {
            if let ItemKind::Impl(Impl { self_ty, .. }) = item.kind;
            if let TyKind::Path(QPath::Resolved(_, item_path)) = self_ty.kind;
            let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args;
            if parameters.as_ref().map_or(true, |params| {
                !params.parenthesized && !params.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
            });
            then {
                StackItem::Check {
                    impl_id: item.def_id,
                    in_body: 0,
                    types_to_skip: std::iter::once(self_ty.hir_id).collect(),
                }
            } else {
                StackItem::NoCheck
            }
        };
        self.stack.push(stack_item);
    }

    fn check_item_post(&mut self, _: &LateContext<'_>, item: &Item<'_>) {
        if !matches!(item.kind, ItemKind::OpaqueTy(_)) {
            self.stack.pop();
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        // We want to skip types in trait `impl`s that aren't declared as `Self` in the trait
        // declaration. The collection of those types is all this method implementation does.
        if_chain! {
            if let ImplItemKind::Fn(FnSig { decl, .. }, ..) = impl_item.kind;
            if let Some(&mut StackItem::Check {
                impl_id,
                ref mut types_to_skip,
                ..
            }) = self.stack.last_mut();
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(impl_id);
            then {
                // `self_ty` is the semantic self type of `impl <trait> for <type>`. This cannot be
                // `Self`.
                let self_ty = impl_trait_ref.self_ty();

                // `trait_method_sig` is the signature of the function, how it is declared in the
                // trait, not in the impl of the trait.
                let trait_method = cx
                    .tcx
                    .associated_items(impl_trait_ref.def_id)
                    .find_by_name_and_kind(cx.tcx, impl_item.ident, AssocKind::Fn, impl_trait_ref.def_id)
                    .expect("impl method matches a trait method");
                let trait_method_sig = cx.tcx.fn_sig(trait_method.def_id);
                let trait_method_sig = cx.tcx.erase_late_bound_regions(trait_method_sig);

                // `impl_inputs_outputs` is an iterator over the types (`hir::Ty`) declared in the
                // implementation of the trait.
                let output_hir_ty = if let FnRetTy::Return(ty) = &decl.output {
                    Some(&**ty)
                } else {
                    None
                };
                let impl_inputs_outputs = decl.inputs.iter().chain(output_hir_ty);

                // `impl_hir_ty` (of type `hir::Ty`) represents the type written in the signature.
                //
                // `trait_sem_ty` (of type `ty::Ty`) is the semantic type for the signature in the
                // trait declaration. This is used to check if `Self` was used in the trait
                // declaration.
                //
                // If `any`where in the `trait_sem_ty` the `self_ty` was used verbatim (as opposed
                // to `Self`), we want to skip linting that type and all subtypes of it. This
                // avoids suggestions to e.g. replace `Vec<u8>` with `Vec<Self>`, in an `impl Trait
                // for u8`, when the trait always uses `Vec<u8>`.
                //
                // See also https://github.com/rust-lang/rust-clippy/issues/2894.
                for (impl_hir_ty, trait_sem_ty) in impl_inputs_outputs.zip(trait_method_sig.inputs_and_output) {
                    if trait_sem_ty.walk(cx.tcx).any(|inner| inner == self_ty.into()) {
                        let mut visitor = SkipTyCollector::default();
                        visitor.visit_ty(impl_hir_ty);
                        types_to_skip.extend(visitor.types_to_skip);
                    }
                }
            }
        }
    }

    fn check_body(&mut self, _: &LateContext<'_>, _: &hir::Body<'_>) {
        // `hir_ty_to_ty` cannot be called in `Body`s or it will panic (sometimes). But in bodies
        // we can use `cx.typeck_results.node_type(..)` to get the `ty::Ty` from a `hir::Ty`.
        // However the `node_type()` method can *only* be called in bodies.
        if let Some(&mut StackItem::Check { ref mut in_body, .. }) = self.stack.last_mut() {
            *in_body = in_body.saturating_add(1);
        }
    }

    fn check_body_post(&mut self, _: &LateContext<'_>, _: &hir::Body<'_>) {
        if let Some(&mut StackItem::Check { ref mut in_body, .. }) = self.stack.last_mut() {
            *in_body = in_body.saturating_sub(1);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>) {
        if_chain! {
            if !hir_ty.span.from_expansion();
            if meets_msrv(self.msrv.as_ref(), &msrvs::TYPE_ALIAS_ENUM_VARIANTS);
            if let Some(&StackItem::Check {
                impl_id,
                in_body,
                ref types_to_skip,
            }) = self.stack.last();
            if let TyKind::Path(QPath::Resolved(_, path)) = hir_ty.kind;
            if !matches!(path.res, Res::SelfTy(..) | Res::Def(DefKind::TyParam, _));
            if !types_to_skip.contains(&hir_ty.hir_id);
            let ty = if in_body > 0 {
                cx.typeck_results().node_type(hir_ty.hir_id)
            } else {
                hir_ty_to_ty(cx.tcx, hir_ty)
            };
            if same_type_and_consts(ty, cx.tcx.type_of(impl_id));
            let hir = cx.tcx.hir();
            // prevents false positive on `#[derive(serde::Deserialize)]`
            if !hir.span(hir.get_parent_node(hir_ty.hir_id)).in_derive_expansion();
            then {
                span_lint(cx, hir_ty.span);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if !expr.span.from_expansion();
            if meets_msrv(self.msrv.as_ref(), &msrvs::TYPE_ALIAS_ENUM_VARIANTS);
            if let Some(&StackItem::Check { impl_id, .. }) = self.stack.last();
            if cx.typeck_results().expr_ty(expr) == cx.tcx.type_of(impl_id);
            then {} else { return; }
        }
        match expr.kind {
            ExprKind::Struct(QPath::Resolved(_, path), ..) => match path.res {
                Res::SelfTy(..) => (),
                Res::Def(DefKind::Variant, _) => lint_path_to_variant(cx, path),
                _ => span_lint(cx, path.span),
            },
            // tuple struct instantiation (`Foo(arg)` or `Enum::Foo(arg)`)
            ExprKind::Call(fun, _) => {
                if let ExprKind::Path(QPath::Resolved(_, path)) = fun.kind {
                    if let Res::Def(DefKind::Ctor(ctor_of, _), ..) = path.res {
                        match ctor_of {
                            CtorOf::Variant => lint_path_to_variant(cx, path),
                            CtorOf::Struct => span_lint(cx, path.span),
                        }
                    }
                }
            },
            // unit enum variants (`Enum::A`)
            ExprKind::Path(QPath::Resolved(_, path)) => lint_path_to_variant(cx, path),
            _ => (),
        }
    }

    extract_msrv_attr!(LateContext);
}

#[derive(Default)]
struct SkipTyCollector {
    types_to_skip: Vec<HirId>,
}

impl<'tcx> Visitor<'tcx> for SkipTyCollector {
    type Map = Map<'tcx>;

    fn visit_infer(&mut self, inf: &hir::InferArg) {
        self.types_to_skip.push(inf.hir_id);

        walk_inf(self, inf);
    }
    fn visit_ty(&mut self, hir_ty: &hir::Ty<'_>) {
        self.types_to_skip.push(hir_ty.hir_id);

        walk_ty(self, hir_ty);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

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

fn lint_path_to_variant(cx: &LateContext<'_>, path: &Path<'_>) {
    if let [.., self_seg, _variant] = path.segments {
        let span = path
            .span
            .with_hi(self_seg.args().span_ext().unwrap_or(self_seg.ident.span).hi());
        span_lint(cx, span);
    }
}
