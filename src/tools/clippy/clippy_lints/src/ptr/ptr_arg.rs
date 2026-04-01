use super::PTR_ARG;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::res::MaybeResPath;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{VEC_METHODS_SHADOWING_SLICE_METHODS, get_expr_use_or_unification_node, is_lint_allowed, sym};
use hir::LifetimeKind;
use rustc_abi::ExternAbi;
use rustc_errors::Applicability;
use rustc_hir::hir_id::{HirId, HirIdMap};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{
    self as hir, AnonConst, BindingMode, Body, Expr, ExprKind, FnSig, GenericArg, Lifetime, Mutability, Node, OwnerId,
    Param, PatKind, QPath, TyKind,
};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_lint::LateContext;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Binder, ClauseKind, ExistentialPredicate, List, PredicateKind, Ty};
use rustc_span::Span;
use rustc_span::symbol::Symbol;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use std::{fmt, iter};

pub(super) fn check_body<'tcx>(
    cx: &LateContext<'tcx>,
    body: &Body<'tcx>,
    item_id: OwnerId,
    sig: &FnSig<'tcx>,
    is_trait_item: bool,
) {
    if !matches!(sig.header.abi, ExternAbi::Rust) {
        // Ignore `extern` functions with non-Rust calling conventions
        return;
    }

    let decl = sig.decl;
    let sig = cx.tcx.fn_sig(item_id).instantiate_identity().skip_binder();
    let lint_args: Vec<_> = check_fn_args(cx, sig, decl.inputs, body.params)
        .filter(|arg| !is_trait_item || arg.mutability() == Mutability::Not)
        .collect();
    let results = check_ptr_arg_usage(cx, body, &lint_args);

    for (result, args) in iter::zip(&results, &lint_args).filter(|(r, _)| !r.skip) {
        span_lint_hir_and_then(cx, PTR_ARG, args.emission_id, args.span, args.build_msg(), |diag| {
            diag.multipart_suggestion(
                "change this to",
                iter::once((args.span, format!("{}{}", args.ref_prefix, args.deref_ty.display(cx))))
                    .chain(result.replacements.iter().map(|r| {
                        (
                            r.expr_span,
                            format!("{}{}", r.self_span.get_source_text(cx).unwrap(), r.replacement),
                        )
                    }))
                    .collect(),
                Applicability::Unspecified,
            );
        });
    }
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, item_id: OwnerId, sig: &FnSig<'tcx>) {
    if !matches!(sig.header.abi, ExternAbi::Rust) {
        // Ignore `extern` functions with non-Rust calling conventions
        return;
    }

    for arg in check_fn_args(
        cx,
        cx.tcx.fn_sig(item_id).instantiate_identity().skip_binder(),
        sig.decl.inputs,
        &[],
    )
    .filter(|arg| arg.mutability() == Mutability::Not)
    {
        span_lint_hir_and_then(cx, PTR_ARG, arg.emission_id, arg.span, arg.build_msg(), |diag| {
            diag.span_suggestion(
                arg.span,
                "change this to",
                format!("{}{}", arg.ref_prefix, arg.deref_ty.display(cx)),
                Applicability::Unspecified,
            );
        });
    }
}

#[derive(Default)]
struct PtrArgResult {
    skip: bool,
    replacements: Vec<PtrArgReplacement>,
}

struct PtrArgReplacement {
    expr_span: Span,
    self_span: Span,
    replacement: &'static str,
}

struct PtrArg<'tcx> {
    idx: usize,
    emission_id: HirId,
    span: Span,
    ty_name: Symbol,
    method_renames: &'static [(Symbol, &'static str)],
    ref_prefix: RefPrefix,
    deref_ty: DerefTy<'tcx>,
}
impl PtrArg<'_> {
    fn build_msg(&self) -> String {
        format!(
            "writing `&{}{}` instead of `&{}{}` involves a new object where a slice will do",
            self.ref_prefix.mutability.prefix_str(),
            self.ty_name,
            self.ref_prefix.mutability.prefix_str(),
            self.deref_ty.argless_str(),
        )
    }

    fn mutability(&self) -> Mutability {
        self.ref_prefix.mutability
    }
}

struct RefPrefix {
    lt: Lifetime,
    mutability: Mutability,
}
impl fmt::Display for RefPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use fmt::Write;
        f.write_char('&')?;
        if !self.lt.is_anonymous() {
            self.lt.ident.fmt(f)?;
            f.write_char(' ')?;
        }
        f.write_str(self.mutability.prefix_str())
    }
}

struct DerefTyDisplay<'a, 'tcx>(&'a LateContext<'tcx>, &'a DerefTy<'tcx>);
impl fmt::Display for DerefTyDisplay<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::Write;
        match self.1 {
            DerefTy::Str => f.write_str("str"),
            DerefTy::Path => f.write_str("Path"),
            DerefTy::Slice(hir_ty, ty) => {
                f.write_char('[')?;
                match hir_ty.and_then(|s| s.get_source_text(self.0)) {
                    Some(s) => f.write_str(&s)?,
                    None => ty.fmt(f)?,
                }
                f.write_char(']')
            },
        }
    }
}

enum DerefTy<'tcx> {
    Str,
    Path,
    Slice(Option<Span>, Ty<'tcx>),
}
impl<'tcx> DerefTy<'tcx> {
    fn ty(&self, cx: &LateContext<'tcx>) -> Ty<'tcx> {
        match *self {
            Self::Str => cx.tcx.types.str_,
            Self::Path => Ty::new_adt(
                cx.tcx,
                cx.tcx.adt_def(cx.tcx.get_diagnostic_item(sym::Path).unwrap()),
                List::empty(),
            ),
            Self::Slice(_, ty) => Ty::new_slice(cx.tcx, ty),
        }
    }

    fn argless_str(&self) -> &'static str {
        match *self {
            Self::Str => "str",
            Self::Path => "Path",
            Self::Slice(..) => "[_]",
        }
    }

    fn display<'a>(&'a self, cx: &'a LateContext<'tcx>) -> DerefTyDisplay<'a, 'tcx> {
        DerefTyDisplay(cx, self)
    }
}

fn check_fn_args<'cx, 'tcx: 'cx>(
    cx: &'cx LateContext<'tcx>,
    fn_sig: ty::FnSig<'tcx>,
    hir_tys: &'tcx [hir::Ty<'tcx>],
    params: &'tcx [Param<'tcx>],
) -> impl Iterator<Item = PtrArg<'tcx>> + 'cx {
    iter::zip(fn_sig.inputs(), hir_tys)
        .enumerate()
        .filter_map(move |(i, (ty, hir_ty))| {
            if let ty::Ref(_, ty, mutability) = *ty.kind()
                && let  ty::Adt(adt, args) = *ty.kind()
                && let TyKind::Ref(lt, ref ty) = hir_ty.kind
                && let TyKind::Path(QPath::Resolved(None, path)) = ty.ty.kind
                // Check that the name as typed matches the actual name of the type.
                // e.g. `fn foo(_: &Foo)` shouldn't trigger the lint when `Foo` is an alias for `Vec`
                && let [.., name] = path.segments
                && cx.tcx.item_name(adt.did()) == name.ident.name
            {
                let emission_id = params.get(i).map_or(hir_ty.hir_id, |param| param.hir_id);
                let (method_renames, deref_ty) = match cx.tcx.get_diagnostic_name(adt.did()) {
                    Some(sym::Vec) => (
                        [(sym::clone, ".to_owned()")].as_slice(),
                        DerefTy::Slice(
                            if let Some(name_args) = name.args
                                && let [GenericArg::Type(ty), ..] = name_args.args
                            {
                                Some(ty.span)
                            } else {
                                None
                            },
                            args.type_at(0),
                        ),
                    ),
                    _ if Some(adt.did()) == cx.tcx.lang_items().string() => (
                        [(sym::clone, ".to_owned()"), (sym::as_str, "")].as_slice(),
                        DerefTy::Str,
                    ),
                    Some(sym::PathBuf) => (
                        [(sym::clone, ".to_path_buf()"), (sym::as_path, "")].as_slice(),
                        DerefTy::Path,
                    ),
                    Some(sym::Cow) if mutability == Mutability::Not => {
                        if let Some(name_args) = name.args
                            && let [GenericArg::Lifetime(lifetime), ty] = name_args.args
                        {
                            if let LifetimeKind::Param(param_def_id) = lifetime.kind
                                && !lifetime.is_anonymous()
                                && fn_sig
                                    .output()
                                    .walk()
                                    .filter_map(ty::GenericArg::as_region)
                                    .filter_map(|lifetime| match lifetime.kind() {
                                        ty::ReEarlyParam(r) => Some(
                                            cx.tcx
                                                .generics_of(cx.tcx.parent(param_def_id.to_def_id()))
                                                .region_param(r, cx.tcx)
                                                .def_id,
                                        ),
                                        ty::ReBound(_, r) => r.kind.get_id(),
                                        ty::ReLateParam(r) => r.kind.get_id(),
                                        ty::ReStatic
                                        | ty::ReVar(_)
                                        | ty::RePlaceholder(_)
                                        | ty::ReErased
                                        | ty::ReError(_) => None,
                                    })
                                    .any(|def_id| def_id.as_local().is_some_and(|def_id| def_id == param_def_id))
                            {
                                // `&Cow<'a, T>` when the return type uses 'a is okay
                                return None;
                            }

                            span_lint_hir_and_then(
                                cx,
                                PTR_ARG,
                                emission_id,
                                hir_ty.span,
                                "using a reference to `Cow` is not recommended",
                                |diag| {
                                    diag.span_suggestion(
                                        hir_ty.span,
                                        "change this to",
                                        match ty.span().get_source_text(cx) {
                                            Some(s) => format!("&{}{s}", mutability.prefix_str()),
                                            None => format!("&{}{}", mutability.prefix_str(), args.type_at(1)),
                                        },
                                        Applicability::Unspecified,
                                    );
                                },
                            );
                        }
                        return None;
                    },
                    _ => return None,
                };
                return Some(PtrArg {
                    idx: i,
                    emission_id,
                    span: hir_ty.span,
                    ty_name: name.ident.name,
                    method_renames,
                    ref_prefix: RefPrefix { lt: *lt, mutability },
                    deref_ty,
                });
            }
            None
        })
}

#[expect(clippy::too_many_lines)]
fn check_ptr_arg_usage<'tcx>(cx: &LateContext<'tcx>, body: &Body<'tcx>, args: &[PtrArg<'tcx>]) -> Vec<PtrArgResult> {
    struct V<'cx, 'tcx> {
        cx: &'cx LateContext<'tcx>,
        /// Map from a local id to which argument it came from (index into `Self::args` and
        /// `Self::results`)
        bindings: HirIdMap<usize>,
        /// The arguments being checked.
        args: &'cx [PtrArg<'tcx>],
        /// The results for each argument (len should match args.len)
        results: Vec<PtrArgResult>,
        /// The number of arguments which can't be linted. Used to return early.
        skip_count: usize,
    }
    impl<'tcx> Visitor<'tcx> for V<'_, 'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;
        fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
            self.cx.tcx
        }

        fn visit_anon_const(&mut self, _: &'tcx AnonConst) {}

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.skip_count == self.args.len() {
                return;
            }

            // Check if this is local we care about
            let Some(&args_idx) = e.res_local_id().and_then(|id| self.bindings.get(&id)) else {
                return walk_expr(self, e);
            };
            let args = &self.args[args_idx];
            let result = &mut self.results[args_idx];

            // Helper function to handle early returns.
            let mut set_skip_flag = || {
                if !result.skip {
                    self.skip_count += 1;
                }
                result.skip = true;
            };

            match get_expr_use_or_unification_node(self.cx.tcx, e) {
                Some((Node::Stmt(_), _)) => (),
                Some((Node::LetStmt(l), _)) => {
                    // Only trace simple bindings. e.g `let x = y;`
                    if let PatKind::Binding(BindingMode::NONE, id, ident, None) = l.pat.kind
                        // Let's not lint for the current parameter. The user may still intend to mutate
                        // (or, if not mutate, then perhaps call a method that's not otherwise available
                        // for) the referenced value behind the parameter through this local let binding
                        // with the underscore being only temporary.
                        && !ident.name.as_str().starts_with('_')
                    {
                        self.bindings.insert(id, args_idx);
                    } else {
                        set_skip_flag();
                    }
                },
                Some((Node::Expr(use_expr), child_id)) => {
                    if let ExprKind::Index(e, ..) = use_expr.kind
                        && e.hir_id == child_id
                    {
                        // Indexing works with both owned and its dereferenced type
                        return;
                    }

                    if let ExprKind::MethodCall(name, receiver, ..) = use_expr.kind
                        && receiver.hir_id == child_id
                    {
                        let name = name.ident.name;

                        // Check if the method can be renamed.
                        if let Some((_, replacement)) = args.method_renames.iter().find(|&&(x, _)| x == name) {
                            result.replacements.push(PtrArgReplacement {
                                expr_span: use_expr.span,
                                self_span: receiver.span,
                                replacement,
                            });
                            return;
                        }

                        // Some methods exist on both `[T]` and `Vec<T>`, such as `len`, where the receiver type
                        // doesn't coerce to a slice and our adjusted type check below isn't enough,
                        // but it would still be valid to call with a slice
                        if VEC_METHODS_SHADOWING_SLICE_METHODS.contains(&name) {
                            return;
                        }
                    }

                    // If the expression's type gets adjusted down to the deref type, we might as
                    // well have started with that deref type -- the lint should fire
                    let deref_ty = args.deref_ty.ty(self.cx);
                    let adjusted_ty = self.cx.typeck_results().expr_ty_adjusted(e).peel_refs();
                    if adjusted_ty == deref_ty {
                        return;
                    }

                    // If the expression's type is constrained by `dyn Trait`, see if the deref
                    // type implements the trait(s) as well, and if so, the lint should fire
                    if let ty::Dynamic(preds, ..) = adjusted_ty.kind()
                        && matches_preds(self.cx, deref_ty, preds)
                    {
                        return;
                    }

                    set_skip_flag();
                },
                _ => set_skip_flag(),
            }
        }
    }

    let mut skip_count = 0;
    let mut results = args.iter().map(|_| PtrArgResult::default()).collect::<Vec<_>>();
    let mut v = V {
        cx,
        bindings: args
            .iter()
            .enumerate()
            .filter_map(|(i, arg)| {
                let param = &body.params[arg.idx];
                match param.pat.kind {
                    PatKind::Binding(BindingMode::NONE, id, ident, None)
                        if !is_lint_allowed(cx, PTR_ARG, param.hir_id)
                            // Let's not lint for the current parameter. The user may still intend to mutate
                            // (or, if not mutate, then perhaps call a method that's not otherwise available
                            // for) the referenced value behind the parameter with the underscore being only
                            // temporary.
                            && !ident.name.as_str().starts_with('_') =>
                    {
                        Some((id, i))
                    },
                    _ => {
                        skip_count += 1;
                        results[i].skip = true;
                        None
                    },
                }
            })
            .collect(),
        args,
        results,
        skip_count,
    };
    v.visit_expr(body.value);
    v.results
}

fn matches_preds<'tcx>(
    cx: &LateContext<'tcx>,
    ty: Ty<'tcx>,
    preds: &'tcx [ty::PolyExistentialPredicate<'tcx>],
) -> bool {
    let infcx = cx.tcx.infer_ctxt().build(cx.typing_mode());
    preds
        .iter()
        .all(|&p| match cx.tcx.instantiate_bound_regions_with_erased(p) {
            ExistentialPredicate::Trait(p) => infcx
                .type_implements_trait(p.def_id, [ty.into()].into_iter().chain(p.args.iter()), cx.param_env)
                .must_apply_modulo_regions(),
            ExistentialPredicate::Projection(p) => infcx.predicate_must_hold_modulo_regions(&Obligation::new(
                cx.tcx,
                ObligationCause::dummy(),
                cx.param_env,
                cx.tcx
                    .mk_predicate(Binder::dummy(PredicateKind::Clause(ClauseKind::Projection(
                        p.with_self_ty(cx.tcx, ty),
                    )))),
            )),
            ExistentialPredicate::AutoTrait(p) => infcx
                .type_implements_trait(p, [ty], cx.param_env)
                .must_apply_modulo_regions(),
        })
}
