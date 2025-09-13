use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::is_from_proc_macro;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::ty::{same_type_and_consts, ty_from_hir_ty};
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir::def::{CtorOf, DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::{InferKind, Visitor, VisitorExt, walk_ty};
use rustc_hir::{
    self as hir, AmbigArg, Expr, ExprKind, FnRetTy, FnSig, GenericArgsParentheses, GenericParam, GenericParamKind,
    HirId, Impl, ImplItemKind, Item, ItemKind, Pat, PatExpr, PatExprKind, PatKind, Path, QPath, Ty, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty as MiddleTy;
use rustc_session::impl_lint_pass;
use rustc_span::Span;

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
    ///
    /// ### Example
    /// ```no_run
    /// struct Foo;
    /// impl Foo {
    ///     fn new() -> Foo {
    ///         Foo {}
    ///     }
    /// }
    /// ```
    /// could be
    /// ```no_run
    /// struct Foo;
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

pub struct UseSelf {
    msrv: Msrv,
    stack: Vec<StackItem>,
}

impl UseSelf {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            msrv: conf.msrv,
            stack: Vec::new(),
        }
    }
}

#[derive(Debug)]
enum StackItem {
    Check {
        impl_id: LocalDefId,
        types_to_skip: FxHashSet<HirId>,
    },
    NoCheck,
}

impl_lint_pass!(UseSelf => [USE_SELF]);

const SEGMENTS_MSG: &str = "segments should be composed of at least 1 element";

impl<'tcx> LateLintPass<'tcx> for UseSelf {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &Item<'tcx>) {
        // We push the self types of `impl`s on a stack here. Only the top type on the stack is
        // relevant for linting, since this is the self type of the `impl` we're currently in. To
        // avoid linting on nested items, we push `StackItem::NoCheck` on the stack to signal, that
        // we're in an `impl` or nested item, that we don't want to lint
        let stack_item = if let ItemKind::Impl(Impl { self_ty, generics, .. }) = item.kind
            && let TyKind::Path(QPath::Resolved(_, item_path)) = self_ty.kind
            && let parameters = &item_path.segments.last().expect(SEGMENTS_MSG).args
            && parameters
                .as_ref()
                .is_none_or(|params| params.parenthesized == GenericArgsParentheses::No)
            && !item.span.from_expansion()
            // expensive, should be last check
            && !is_from_proc_macro(cx, item)
        {
            // Self cannot be used inside const generic parameters
            let types_to_skip = generics
                .params
                .iter()
                .filter_map(|param| match param {
                    GenericParam {
                        kind:
                            GenericParamKind::Const {
                                ty: Ty { hir_id, .. }, ..
                            },
                        ..
                    } => Some(*hir_id),
                    _ => None,
                })
                .chain(std::iter::once(self_ty.hir_id))
                .collect();
            StackItem::Check {
                impl_id: item.owner_id.def_id,
                types_to_skip,
            }
        } else {
            StackItem::NoCheck
        };
        self.stack.push(stack_item);
    }

    fn check_item_post(&mut self, _: &LateContext<'_>, _: &Item<'_>) {
        self.stack.pop();
    }

    fn check_impl_item(&mut self, cx: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        // Checking items of `impl Self` blocks in which macro expands into.
        if impl_item.span.from_expansion() {
            self.stack.push(StackItem::NoCheck);
            return;
        }
        // We want to skip types in trait `impl`s that aren't declared as `Self` in the trait
        // declaration. The collection of those types is all this method implementation does.
        if let ImplItemKind::Fn(FnSig { decl, .. }, ..) = impl_item.kind
            && let Some(&mut StackItem::Check {
                impl_id,
                ref mut types_to_skip,
                ..
            }) = self.stack.last_mut()
            && let Some(impl_trait_ref) = cx.tcx.impl_trait_ref(impl_id)
        {
            // `self_ty` is the semantic self type of `impl <trait> for <type>`. This cannot be
            // `Self`.
            let self_ty = impl_trait_ref.instantiate_identity().self_ty();

            // `trait_method_sig` is the signature of the function, how it is declared in the
            // trait, not in the impl of the trait.
            let trait_method = cx
                .tcx
                .trait_item_of(impl_item.owner_id)
                .expect("impl method matches a trait method");
            let trait_method_sig = cx.tcx.fn_sig(trait_method).instantiate_identity();
            let trait_method_sig = cx.tcx.instantiate_bound_regions_with_erased(trait_method_sig);

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
                    visitor.visit_ty_unambig(impl_hir_ty);
                    types_to_skip.extend(visitor.types_to_skip);
                }
            }
        }
    }

    fn check_impl_item_post(&mut self, _: &LateContext<'_>, impl_item: &hir::ImplItem<'_>) {
        if impl_item.span.from_expansion()
            && let Some(StackItem::NoCheck) = self.stack.last()
        {
            self.stack.pop();
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, hir_ty: &Ty<'tcx, AmbigArg>) {
        if !hir_ty.span.from_expansion()
            && let Some(&StackItem::Check {
                impl_id,
                ref types_to_skip,
            }) = self.stack.last()
            && let TyKind::Path(QPath::Resolved(_, path)) = hir_ty.kind
            && !matches!(
                path.res,
                Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } | Res::Def(DefKind::TyParam, _)
            )
            && !types_to_skip.contains(&hir_ty.hir_id)
            && let ty = ty_from_hir_ty(cx, hir_ty.as_unambig_ty())
            && let impl_ty = cx.tcx.type_of(impl_id).instantiate_identity()
            && same_type_and_consts(ty, impl_ty)
            // Ensure the type we encounter and the one from the impl have the same lifetime parameters. It may be that
            // the lifetime parameters of `ty` are elided (`impl<'a> Foo<'a> { fn new() -> Self { Foo{..} } }`, in
            // which case we must still trigger the lint.
            && (has_no_lifetime(ty) || same_lifetimes(ty, impl_ty))
            && self.msrv.meets(cx, msrvs::TYPE_ALIAS_ENUM_VARIANTS)
        {
            span_lint(cx, hir_ty.span);
        }
    }

    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if !expr.span.from_expansion()
            && let Some(&StackItem::Check { impl_id, .. }) = self.stack.last()
            && cx.typeck_results().expr_ty(expr) == cx.tcx.type_of(impl_id).instantiate_identity()
            && self.msrv.meets(cx, msrvs::TYPE_ALIAS_ENUM_VARIANTS)
        {
        } else {
            return;
        }
        match expr.kind {
            ExprKind::Struct(QPath::Resolved(_, path), ..) => check_path(cx, path),
            ExprKind::Call(fun, _) => {
                if let ExprKind::Path(QPath::Resolved(_, path)) = fun.kind {
                    check_path(cx, path);
                }
            },
            ExprKind::Path(QPath::Resolved(_, path)) => check_path(cx, path),
            _ => (),
        }
    }

    fn check_pat(&mut self, cx: &LateContext<'_>, pat: &Pat<'_>) {
        if !pat.span.from_expansion()
            && let Some(&StackItem::Check { impl_id, .. }) = self.stack.last()
            // get the path from the pattern
            && let PatKind::Expr(&PatExpr { kind: PatExprKind::Path(QPath::Resolved(_, path)), .. })
                 | PatKind::TupleStruct(QPath::Resolved(_, path), _, _)
                 | PatKind::Struct(QPath::Resolved(_, path), _, _) = pat.kind
            && cx.typeck_results().pat_ty(pat) == cx.tcx.type_of(impl_id).instantiate_identity()
            && self.msrv.meets(cx, msrvs::TYPE_ALIAS_ENUM_VARIANTS)
        {
            check_path(cx, path);
        }
    }
}

#[derive(Default)]
struct SkipTyCollector {
    types_to_skip: Vec<HirId>,
}

impl Visitor<'_> for SkipTyCollector {
    fn visit_infer(&mut self, inf_id: HirId, _inf_span: Span, kind: InferKind<'_>) -> Self::Result {
        // Conservatively assume ambiguously kinded inferred arguments are type arguments
        if let InferKind::Ambig(_) | InferKind::Ty(_) = kind {
            self.types_to_skip.push(inf_id);
        }
        self.visit_id(inf_id);
    }
    fn visit_ty(&mut self, hir_ty: &Ty<'_, AmbigArg>) {
        self.types_to_skip.push(hir_ty.hir_id);

        walk_ty(self, hir_ty);
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

fn check_path(cx: &LateContext<'_>, path: &Path<'_>) {
    match path.res {
        Res::Def(DefKind::Ctor(CtorOf::Variant, _) | DefKind::Variant, ..) => {
            lint_path_to_variant(cx, path);
        },
        Res::Def(DefKind::Ctor(CtorOf::Struct, _) | DefKind::Struct, ..) => span_lint(cx, path.span),
        _ => (),
    }
}

fn lint_path_to_variant(cx: &LateContext<'_>, path: &Path<'_>) {
    if let [.., self_seg, _variant] = path.segments {
        let span = path
            .span
            .with_hi(self_seg.args().span_ext().unwrap_or(self_seg.ident.span).hi());
        span_lint(cx, span);
    }
}

/// Returns `true` if types `a` and `b` have the same lifetime parameters, otherwise returns
/// `false`.
///
/// This function does not check that types `a` and `b` are the same types.
fn same_lifetimes<'tcx>(a: MiddleTy<'tcx>, b: MiddleTy<'tcx>) -> bool {
    use rustc_middle::ty::{Adt, GenericArgKind};
    match (&a.kind(), &b.kind()) {
        (&Adt(_, args_a), &Adt(_, args_b)) => {
            args_a
                .iter()
                .zip(args_b.iter())
                .all(|(arg_a, arg_b)| match (arg_a.kind(), arg_b.kind()) {
                    // TODO: Handle inferred lifetimes
                    (GenericArgKind::Lifetime(inner_a), GenericArgKind::Lifetime(inner_b)) => inner_a == inner_b,
                    (GenericArgKind::Type(type_a), GenericArgKind::Type(type_b)) => same_lifetimes(type_a, type_b),
                    _ => true,
                })
        },
        _ => a == b,
    }
}

/// Returns `true` if `ty` has no lifetime parameter, otherwise returns `false`.
fn has_no_lifetime(ty: MiddleTy<'_>) -> bool {
    use rustc_middle::ty::{Adt, GenericArgKind};
    match ty.kind() {
        &Adt(_, args) => !args
            .iter()
            // TODO: Handle inferred lifetimes
            .any(|arg| matches!(arg.kind(), GenericArgKind::Lifetime(..))),
        _ => true,
    }
}
