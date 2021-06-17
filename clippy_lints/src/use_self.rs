use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::ty::same_type_and_consts;
use clippy_utils::{in_macro, meets_msrv, msrvs};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{
    self as hir,
    def::{CtorOf, DefKind, Res},
    def_id::LocalDefId,
    intravisit::{walk_ty, NestedVisitorMap, Visitor},
    Expr, ExprKind, FnRetTy, FnSig, GenericArg, HirId, Impl, ImplItemKind, Item, ItemKind, Node, Path, QPath, TyKind,
};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::{AssocKind, Ty};
use rustc_semver::RustcVersion;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::Span;
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
    /// - Unaddressed false negative in fn bodies of trait implementations
    /// - False positive with assotiated types in traits (#4140)
    ///
    /// **Example:**
    ///
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
        hir_id: HirId,
        impl_trait_ref_def_id: Option<LocalDefId>,
        types_to_skip: Vec<HirId>,
        types_to_lint: Vec<HirId>,
    },
    NoCheck,
}

impl_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

impl<'tcx> LateLintPass<'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'_>, item: &Item<'_>) {
        if !is_item_interesting(item) {
            // This does two things:
            //  1) Reduce needless churn on `self.stack`
            //  2) Don't push `StackItem::NoCheck` when entering `ItemKind::OpaqueTy`,
            //     in order to lint `foo() -> impl <..>`
            return;
        }
        // We push the self types of `impl`s on a stack here. Only the top type on the stack is
        // relevant for linting, since this is the self type of the `impl` we're currently in. To
        // avoid linting on nested items, we push `StackItem::NoCheck` on the stack to signal, that
        // we're in an `impl` or nested item, that we don't want to lint
        let stack_item = if_chain! {
            if let ItemKind::Impl(Impl { self_ty, ref of_trait, .. }) = item.kind;
            if let TyKind::Path(QPath::Resolved(_, item_path)) = self_ty.kind;
            let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args;
            if parameters.as_ref().map_or(true, |params| {
                !params.parenthesized && !params.args.iter().any(|arg| matches!(arg, GenericArg::Lifetime(_)))
            });
            then {
                let impl_trait_ref_def_id = of_trait.as_ref().map(|_| cx.tcx.hir().local_def_id(item.hir_id()));
                StackItem::Check {
                    hir_id: self_ty.hir_id,
                    impl_trait_ref_def_id,
                    types_to_lint: Vec::new(),
                    types_to_skip: Vec::new(),
                }
            } else {
                StackItem::NoCheck
            }
        };
        self.stack.push(stack_item);
    }

    fn check_item_post(&mut self, _: &LateContext<'_>, item: &Item<'_>) {
        if is_item_interesting(item) {
            self.stack.pop();
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        // We want to skip types in trait `impl`s that aren't declared as `Self` in the trait
        // declaration. The collection of those types is all this method implementation does.
        if_chain! {
            if let ImplItemKind::Fn(FnSig { decl, .. }, ..) = impl_item.kind;
            if let Some(&mut StackItem::Check {
                impl_trait_ref_def_id: Some(def_id),
                ref mut types_to_skip,
                ..
            }) = self.stack.last_mut();
            if let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(def_id);
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
                    if trait_sem_ty.walk().any(|inner| inner == self_ty.into()) {
                        let mut visitor = SkipTyCollector::default();
                        visitor.visit_ty(impl_hir_ty);
                        types_to_skip.extend(visitor.types_to_skip);
                    }
                }
            }
        }
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &'tcx hir::Body<'_>) {
        // `hir_ty_to_ty` cannot be called in `Body`s or it will panic (sometimes). But in bodies
        // we can use `cx.typeck_results.node_type(..)` to get the `ty::Ty` from a `hir::Ty`.
        // However the `node_type()` method can *only* be called in bodies.
        //
        // This method implementation determines which types should get linted in a `Body` and
        // which shouldn't, with a visitor. We could directly lint in the visitor, but then we
        // could only allow this lint on item scope. And we would have to check if those types are
        // already dealt with in `check_ty` anyway.
        if let Some(StackItem::Check {
            hir_id,
            types_to_lint,
            types_to_skip,
            ..
        }) = self.stack.last_mut()
        {
            let self_ty = ty_from_hir_id(cx, *hir_id);

            let mut visitor = LintTyCollector {
                cx,
                self_ty,
                types_to_lint: vec![],
                types_to_skip: vec![],
            };
            visitor.visit_expr(&body.value);
            types_to_lint.extend(visitor.types_to_lint);
            types_to_skip.extend(visitor.types_to_skip);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'_>, hir_ty: &hir::Ty<'_>) {
        if_chain! {
            if !in_macro(hir_ty.span) && !in_impl(cx, hir_ty);
            if meets_msrv(self.msrv.as_ref(), &msrvs::TYPE_ALIAS_ENUM_VARIANTS);
            if let Some(StackItem::Check {
                hir_id,
                types_to_lint,
                types_to_skip,
                ..
            }) = self.stack.last();
            if !types_to_skip.contains(&hir_ty.hir_id);
            if types_to_lint.contains(&hir_ty.hir_id)
                || {
                    let self_ty = ty_from_hir_id(cx, *hir_id);
                    should_lint_ty(hir_ty, hir_ty_to_ty(cx.tcx, hir_ty), self_ty)
                };
            let hir = cx.tcx.hir();
            let id = hir.get_parent_node(hir_ty.hir_id);
            if !hir.opt_span(id).map_or(false, in_macro);
            then {
                span_lint(cx, hir_ty.span);
            }
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if_chain! {
            if !in_macro(expr.span);
            if meets_msrv(self.msrv.as_ref(), &msrvs::TYPE_ALIAS_ENUM_VARIANTS);
            if let Some(StackItem::Check { hir_id, .. }) = self.stack.last();
            if cx.typeck_results().expr_ty(expr) == ty_from_hir_id(cx, *hir_id);
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

    fn visit_ty(&mut self, hir_ty: &hir::Ty<'_>) {
        self.types_to_skip.push(hir_ty.hir_id);

        walk_ty(self, hir_ty);
    }

    fn nested_visit_map(&mut self) -> NestedVisitorMap<Self::Map> {
        NestedVisitorMap::None
    }
}

struct LintTyCollector<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    self_ty: Ty<'tcx>,
    types_to_lint: Vec<HirId>,
    types_to_skip: Vec<HirId>,
}

impl<'a, 'tcx> Visitor<'tcx> for LintTyCollector<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_ty(&mut self, hir_ty: &'tcx hir::Ty<'_>) {
        if_chain! {
            if let Some(ty) = self.cx.typeck_results().node_type_opt(hir_ty.hir_id);
            if should_lint_ty(hir_ty, ty, self.self_ty);
            then {
                self.types_to_lint.push(hir_ty.hir_id);
            } else {
                self.types_to_skip.push(hir_ty.hir_id);
            }
        }

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

fn is_item_interesting(item: &Item<'_>) -> bool {
    use rustc_hir::ItemKind::{Const, Enum, Fn, Impl, Static, Struct, Trait, Union};
    matches!(
        item.kind,
        Impl { .. } | Static(..) | Const(..) | Fn(..) | Enum(..) | Struct(..) | Union(..) | Trait(..)
    )
}

fn ty_from_hir_id<'tcx>(cx: &LateContext<'tcx>, hir_id: HirId) -> Ty<'tcx> {
    if let Some(Node::Ty(hir_ty)) = cx.tcx.hir().find(hir_id) {
        hir_ty_to_ty(cx.tcx, hir_ty)
    } else {
        unreachable!("This function should only be called with `HirId`s that are for sure `Node::Ty`")
    }
}

fn in_impl(cx: &LateContext<'tcx>, hir_ty: &hir::Ty<'_>) -> bool {
    let map = cx.tcx.hir();
    let parent = map.get_parent_node(hir_ty.hir_id);
    if_chain! {
        if let Some(Node::Item(item)) = map.find(parent);
        if let ItemKind::Impl { .. } = item.kind;
        then {
            true
        } else {
            false
        }
    }
}

fn should_lint_ty(hir_ty: &hir::Ty<'_>, ty: Ty<'_>, self_ty: Ty<'_>) -> bool {
    if_chain! {
        if same_type_and_consts(ty, self_ty);
        if let TyKind::Path(QPath::Resolved(_, path)) = hir_ty.kind;
        then {
            !matches!(path.res, Res::SelfTy(..) | Res::Def(DefKind::TyParam, _))
        } else {
            false
        }
    }
}
