use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::res::MaybeDef;
use clippy_utils::{fulfill_or_allowed, get_parent_as_impl, sym};
use rustc_hir::def::Res;
use rustc_hir::def_id::{DefId, DefIdSet};
use rustc_hir::{
    FnRetTy, GenericArg, GenericBound, HirId, ImplItem, ImplItemKind, ImplicitSelfKind, Item, ItemKind, Mutability,
    Node, OpaqueTyOrigin, PathSegment, PrimTy, QPath, TraitItemId, TyKind,
};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::{self, FnSig, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::kw;
use rustc_span::{Ident, Span, Symbol};
use rustc_trait_selection::traits::supertrait_def_ids;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for items that implement `.len()` but not
    /// `.is_empty()`.
    ///
    /// ### Why is this bad?
    /// It is good custom to have both methods, because for
    /// some data structures, asking about the length will be a costly operation,
    /// whereas `.is_empty()` can usually answer in constant time. Also it used to
    /// lead to false positives on the [`len_zero`](#len_zero) lint – currently that
    /// lint will ignore such entities.
    ///
    /// ### Example
    /// ```ignore
    /// impl X {
    ///     pub fn len(&self) -> usize {
    ///         ..
    ///     }
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub LEN_WITHOUT_IS_EMPTY,
    style,
    "traits or impls with a public `len` method but no corresponding `is_empty` method"
}

declare_lint_pass!(LenWithoutIsEmpty => [LEN_WITHOUT_IS_EMPTY]);

impl<'tcx> LateLintPass<'tcx> for LenWithoutIsEmpty {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx Item<'_>) {
        if let ItemKind::Trait(_, _, _, ident, _, _, trait_items) = item.kind
            && !item.span.from_expansion()
        {
            check_trait_items(cx, item, ident, trait_items);
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx ImplItem<'_>) {
        if item.ident.name == sym::len
            && let ImplItemKind::Fn(sig, _) = &item.kind
            && sig.decl.implicit_self.has_implicit_self()
            && sig.decl.inputs.len() == 1
            && cx.effective_visibilities.is_exported(item.owner_id.def_id)
            && matches!(sig.decl.output, FnRetTy::Return(_))
            && let Some(imp) = get_parent_as_impl(cx.tcx, item.hir_id())
            && imp.of_trait.is_none()
            && let TyKind::Path(ty_path) = &imp.self_ty.kind
            && let Some(ty_id) = cx.qpath_res(ty_path, imp.self_ty.hir_id).opt_def_id()
            && let Some(local_id) = ty_id.as_local()
            && let ty_hir_id = cx.tcx.local_def_id_to_hir_id(local_id)
            && let Some(output) = LenOutput::new(cx, cx.tcx.fn_sig(item.owner_id).instantiate_identity().skip_binder())
        {
            let (name, kind) = match cx.tcx.hir_node(ty_hir_id) {
                Node::ForeignItem(x) => (x.ident.name, "extern type"),
                Node::Item(x) => match x.kind {
                    ItemKind::Struct(ident, ..) => (ident.name, "struct"),
                    ItemKind::Enum(ident, ..) => (ident.name, "enum"),
                    ItemKind::Union(ident, ..) => (ident.name, "union"),
                    _ => (x.kind.ident().unwrap().name, "type"),
                },
                _ => return,
            };
            check_for_is_empty(
                cx,
                sig.span,
                sig.decl.implicit_self,
                output,
                ty_id,
                name,
                kind,
                item.hir_id(),
                ty_hir_id,
            );
        }
    }
}

fn check_trait_items(cx: &LateContext<'_>, visited_trait: &Item<'_>, ident: Ident, trait_items: &[TraitItemId]) {
    fn is_named_self(cx: &LateContext<'_>, item: TraitItemId, name: Symbol) -> bool {
        cx.tcx.item_name(item.owner_id) == name
            && matches!(
                cx.tcx.fn_arg_idents(item.owner_id),
                [Some(Ident {
                    name: kw::SelfLower,
                    ..
                })],
            )
    }

    // fill the set with current and super traits
    fn fill_trait_set(traitt: DefId, set: &mut DefIdSet, cx: &LateContext<'_>) {
        if set.insert(traitt) {
            for supertrait in supertrait_def_ids(cx.tcx, traitt) {
                fill_trait_set(supertrait, set, cx);
            }
        }
    }

    if cx.effective_visibilities.is_exported(visited_trait.owner_id.def_id)
        && trait_items.iter().any(|&i| is_named_self(cx, i, sym::len))
    {
        let mut current_and_super_traits = DefIdSet::default();
        fill_trait_set(visited_trait.owner_id.to_def_id(), &mut current_and_super_traits, cx);
        let is_empty_method_found = current_and_super_traits
            .items()
            .flat_map(|&i| cx.tcx.associated_items(i).filter_by_name_unhygienic(sym::is_empty))
            .any(|i| i.is_method() && cx.tcx.fn_sig(i.def_id).skip_binder().inputs().skip_binder().len() == 1);

        if !is_empty_method_found {
            span_lint(
                cx,
                LEN_WITHOUT_IS_EMPTY,
                visited_trait.span,
                format!(
                    "trait `{}` has a `len` method but no (possibly inherited) `is_empty` method",
                    ident.name
                ),
            );
        }
    }
}

fn extract_future_output<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>) -> Option<&'tcx PathSegment<'tcx>> {
    if let ty::Alias(_, alias_ty) = ty.kind()
        && let Some(Node::OpaqueTy(opaque)) = cx.tcx.hir_get_if_local(alias_ty.def_id)
        && let OpaqueTyOrigin::AsyncFn { .. } = opaque.origin
        && let [GenericBound::Trait(trait_ref)] = &opaque.bounds
        && let Some(segment) = trait_ref.trait_ref.path.segments.last()
        && let Some(generic_args) = segment.args
        && let [constraint] = generic_args.constraints
        && let Some(ty) = constraint.ty()
        && let TyKind::Path(QPath::Resolved(_, path)) = ty.kind
        && let [segment] = path.segments
    {
        return Some(segment);
    }

    None
}

fn is_first_generic_integral<'tcx>(segment: &'tcx PathSegment<'tcx>) -> bool {
    if let Some(generic_args) = segment.args
        && let [GenericArg::Type(ty), ..] = &generic_args.args
        && let TyKind::Path(QPath::Resolved(_, path)) = ty.kind
        && let [segment, ..] = &path.segments
        && matches!(segment.res, Res::PrimTy(PrimTy::Uint(_) | PrimTy::Int(_)))
    {
        true
    } else {
        false
    }
}

#[derive(Debug, Clone, Copy)]
enum LenOutput {
    Integral,
    Option(DefId),
    Result(DefId),
}

impl LenOutput {
    fn new<'tcx>(cx: &LateContext<'tcx>, sig: FnSig<'tcx>) -> Option<Self> {
        if let Some(segment) = extract_future_output(cx, sig.output()) {
            let res = segment.res;

            if matches!(res, Res::PrimTy(PrimTy::Uint(_) | PrimTy::Int(_))) {
                return Some(Self::Integral);
            }

            if let Res::Def(_, def_id) = res
                && let Some(res) = match cx.tcx.get_diagnostic_name(def_id) {
                    Some(sym::Option) => Some(Self::Option(def_id)),
                    Some(sym::Result) => Some(Self::Result(def_id)),
                    _ => None,
                }
                && is_first_generic_integral(segment)
            {
                return Some(res);
            }

            return None;
        }

        match *sig.output().kind() {
            ty::Int(_) | ty::Uint(_) => Some(Self::Integral),
            ty::Adt(adt, subs) => match cx.tcx.get_diagnostic_name(adt.did()) {
                Some(sym::Option) => subs.type_at(0).is_integral().then(|| Self::Option(adt.did())),
                Some(sym::Result) => subs.type_at(0).is_integral().then(|| Self::Result(adt.did())),
                _ => None,
            },
            _ => None,
        }
    }

    fn matches_is_empty_output<'tcx>(self, cx: &LateContext<'tcx>, is_empty_output: Ty<'tcx>) -> bool {
        if let Some(segment) = extract_future_output(cx, is_empty_output) {
            return match (self, segment.res) {
                (_, Res::PrimTy(PrimTy::Bool)) => true,
                (Self::Option(_), Res::Def(_, def_id)) if cx.tcx.is_diagnostic_item(sym::Option, def_id) => true,
                (Self::Result(_), Res::Def(_, def_id)) if cx.tcx.is_diagnostic_item(sym::Result, def_id) => true,
                _ => false,
            };
        }

        match (self, is_empty_output.kind()) {
            (_, &ty::Bool) => true,
            (Self::Option(id), &ty::Adt(adt, subs)) if id == adt.did() => subs.type_at(0).is_bool(),
            (Self::Result(id), &ty::Adt(adt, subs)) if id == adt.did() => subs.type_at(0).is_bool(),
            _ => false,
        }
    }
}

/// The expected signature of `is_empty`, based on that of `len`
fn expected_is_empty_sig(len_output: LenOutput, len_self_kind: ImplicitSelfKind) -> String {
    let self_ref = match len_self_kind {
        ImplicitSelfKind::RefImm => "&",
        ImplicitSelfKind::RefMut => "&(mut) ",
        _ => "",
    };
    match len_output {
        LenOutput::Integral => format!("expected signature: `({self_ref}self) -> bool`"),
        LenOutput::Option(_) => {
            format!("expected signature: `({self_ref}self) -> bool` or `({self_ref}self) -> Option<bool>")
        },
        LenOutput::Result(..) => {
            format!("expected signature: `({self_ref}self) -> bool` or `({self_ref}self) -> Result<bool>")
        },
    }
}

/// Checks if the given signature matches the expectations for `is_empty`
fn check_is_empty_sig<'tcx>(
    cx: &LateContext<'tcx>,
    is_empty_sig: FnSig<'tcx>,
    len_self_kind: ImplicitSelfKind,
    len_output: LenOutput,
) -> bool {
    if let [is_empty_self_arg, is_empty_output] = &**is_empty_sig.inputs_and_output
        && len_output.matches_is_empty_output(cx, *is_empty_output)
    {
        match (is_empty_self_arg.kind(), len_self_kind) {
            // if `len` takes `&self`, `is_empty` should do so as well
            (ty::Ref(_, _, Mutability::Not), ImplicitSelfKind::RefImm)
            // if `len` takes `&mut self`, `is_empty` may take that _or_ `&self` (#16190)
            | (ty::Ref(_, _, Mutability::Mut | Mutability::Not), ImplicitSelfKind::RefMut) => true,
            // if len takes `self`, `is_empty` should do so as well
            // XXX: we might want to relax this to allow `&self` and `&mut self`
            (_, ImplicitSelfKind::Imm | ImplicitSelfKind::Mut) if !is_empty_self_arg.is_ref() => true,
            _ => false,
        }
    } else {
        false
    }
}

/// Checks if the given type has an `is_empty` method with the appropriate signature.
#[expect(clippy::too_many_arguments)]
fn check_for_is_empty(
    cx: &LateContext<'_>,
    len_span: Span,
    len_self_kind: ImplicitSelfKind,
    len_output: LenOutput,
    impl_ty: DefId,
    item_name: Symbol,
    item_kind: &str,
    len_method_hir_id: HirId,
    ty_decl_hir_id: HirId,
) {
    // Implementor may be a type alias, in which case we need to get the `DefId` of the aliased type to
    // find the correct inherent impls.
    let impl_ty = if let Some(adt) = cx.tcx.type_of(impl_ty).skip_binder().ty_adt_def() {
        adt.did()
    } else {
        return;
    };

    let is_empty = cx
        .tcx
        .inherent_impls(impl_ty)
        .iter()
        .flat_map(|&id| cx.tcx.associated_items(id).filter_by_name_unhygienic(sym::is_empty))
        .find(|item| item.is_fn());

    let (msg, is_empty_span, is_empty_expected_sig) = match is_empty {
        None => (
            format!("{item_kind} `{item_name}` has a public `len` method, but no `is_empty` method"),
            None,
            None,
        ),
        Some(is_empty) if !cx.effective_visibilities.is_exported(is_empty.def_id.expect_local()) => (
            format!("{item_kind} `{item_name}` has a public `len` method, but a private `is_empty` method"),
            Some(cx.tcx.def_span(is_empty.def_id)),
            None,
        ),
        Some(is_empty)
            if !(is_empty.is_method()
                && check_is_empty_sig(
                    cx,
                    cx.tcx.fn_sig(is_empty.def_id).instantiate_identity().skip_binder(),
                    len_self_kind,
                    len_output,
                )) =>
        {
            (
                format!(
                    "{item_kind} `{item_name}` has a public `len` method, but the `is_empty` method has an unexpected signature",
                ),
                Some(cx.tcx.def_span(is_empty.def_id)),
                Some(expected_is_empty_sig(len_output, len_self_kind)),
            )
        },
        Some(_) => return,
    };

    if !fulfill_or_allowed(cx, LEN_WITHOUT_IS_EMPTY, [len_method_hir_id, ty_decl_hir_id]) {
        span_lint_and_then(cx, LEN_WITHOUT_IS_EMPTY, len_span, msg, |db| {
            if let Some(span) = is_empty_span {
                db.span_note(span, "`is_empty` defined here");
            }
            if let Some(expected_sig) = is_empty_expected_sig {
                db.note(expected_sig);
            }
        });
    }
}
