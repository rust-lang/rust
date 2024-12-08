use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::visitors::contains_unsafe_block;
use clippy_utils::{get_expr_use_or_unification_node, is_lint_allowed, path_def_id, path_to_local, std_or_core};
use hir::LifetimeName;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::hir_id::{HirId, HirIdMap};
use rustc_hir::intravisit::{Visitor, walk_expr};
use rustc_hir::{
    self as hir, AnonConst, BinOpKind, BindingMode, Body, Expr, ExprKind, FnRetTy, FnSig, GenericArg, ImplItemKind,
    ItemKind, Lifetime, Mutability, Node, Param, PatKind, QPath, Safety, TraitFn, TraitItem, TraitItemKind, TyKind,
};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Binder, ClauseKind, ExistentialPredicate, List, PredicateKind, Ty};
use rustc_session::declare_lint_pass;
use rustc_span::symbol::Symbol;
use rustc_span::{Span, sym};
use rustc_target::spec::abi::Abi;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use std::{fmt, iter};

use crate::vec::is_allowed_vec_method;

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for function arguments of type `&String`, `&Vec`,
    /// `&PathBuf`, and `Cow<_>`. It will also suggest you replace `.clone()` calls
    /// with the appropriate `.to_owned()`/`to_string()` calls.
    ///
    /// ### Why is this bad?
    /// Requiring the argument to be of the specific type
    /// makes the function less useful for no benefit; slices in the form of `&[T]`
    /// or `&str` usually suffice and can be obtained from other types, too.
    ///
    /// ### Known problems
    /// There may be `fn(&Vec)`-typed references pointing to your function.
    /// If you have them, you will get a compiler error after applying this lint's
    /// suggestions. You then have the choice to undo your changes or change the
    /// type of the reference.
    ///
    /// Note that if the function is part of your public interface, there may be
    /// other crates referencing it, of which you may not be aware. Carefully
    /// deprecate the function before applying the lint suggestions in this case.
    ///
    /// ### Example
    /// ```ignore
    /// fn foo(&Vec<u32>) { .. }
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// fn foo(&[u32]) { .. }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub PTR_ARG,
    style,
    "fn arguments of the type `&Vec<...>` or `&String`, suggesting to use `&[...]` or `&str` instead, respectively"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for equality comparisons with `ptr::null`
    ///
    /// ### Why is this bad?
    /// It's easier and more readable to use the inherent
    /// `.is_null()`
    /// method instead
    ///
    /// ### Example
    /// ```rust,ignore
    /// use std::ptr;
    ///
    /// if x == ptr::null {
    ///     // ..
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// if x.is_null() {
    ///     // ..
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub CMP_NULL,
    style,
    "comparing a pointer to a null pointer, suggesting to use `.is_null()` instead"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for functions that take immutable references and return
    /// mutable ones. This will not trigger if no unsafe code exists as there
    /// are multiple safe functions which will do this transformation
    ///
    /// To be on the conservative side, if there's at least one mutable
    /// reference with the output lifetime, this lint will not trigger.
    ///
    /// ### Why is this bad?
    /// Creating a mutable reference which can be repeatably derived from an
    /// immutable reference is unsound as it allows creating multiple live
    /// mutable references to the same object.
    ///
    /// This [error](https://github.com/rust-lang/rust/issues/39465) actually
    /// lead to an interim Rust release 1.15.1.
    ///
    /// ### Known problems
    /// This pattern is used by memory allocators to allow allocating multiple
    /// objects while returning mutable references to each one. So long as
    /// different mutable references are returned each time such a function may
    /// be safe.
    ///
    /// ### Example
    /// ```ignore
    /// fn foo(&Foo) -> &mut Bar { .. }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub MUT_FROM_REF,
    correctness,
    "fns that create mutable refs from immutable ref args"
}

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for invalid usages of `ptr::null`.
    ///
    /// ### Why is this bad?
    /// This causes undefined behavior.
    ///
    /// ### Example
    /// ```ignore
    /// // Undefined behavior
    /// unsafe { std::slice::from_raw_parts(ptr::null(), 0); }
    /// ```
    ///
    /// Use instead:
    /// ```ignore
    /// unsafe { std::slice::from_raw_parts(NonNull::dangling().as_ptr(), 0); }
    /// ```
    #[clippy::version = "1.53.0"]
    pub INVALID_NULL_PTR_USAGE,
    correctness,
    "invalid usage of a null pointer, suggesting `NonNull::dangling()` instead"
}

declare_lint_pass!(Ptr => [PTR_ARG, CMP_NULL, MUT_FROM_REF, INVALID_NULL_PTR_USAGE]);

impl<'tcx> LateLintPass<'tcx> for Ptr {
    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx TraitItem<'_>) {
        if let TraitItemKind::Fn(sig, trait_method) = &item.kind {
            if matches!(trait_method, TraitFn::Provided(_)) {
                // Handled by check body.
                return;
            }

            check_mut_from_ref(cx, sig, None);

            if !matches!(sig.header.abi, Abi::Rust) {
                // Ignore `extern` functions with non-Rust calling conventions
                return;
            }

            for arg in check_fn_args(
                cx,
                cx.tcx.fn_sig(item.owner_id).instantiate_identity().skip_binder(),
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
    }

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &Body<'tcx>) {
        let hir = cx.tcx.hir();
        let mut parents = hir.parent_iter(body.value.hir_id);
        let (item_id, sig, is_trait_item) = match parents.next() {
            Some((_, Node::Item(i))) => {
                if let ItemKind::Fn(sig, ..) = &i.kind {
                    (i.owner_id, sig, false)
                } else {
                    return;
                }
            },
            Some((_, Node::ImplItem(i))) => {
                if !matches!(parents.next(),
                    Some((_, Node::Item(i))) if matches!(&i.kind, ItemKind::Impl(i) if i.of_trait.is_none())
                ) {
                    return;
                }
                if let ImplItemKind::Fn(sig, _) = &i.kind {
                    (i.owner_id, sig, false)
                } else {
                    return;
                }
            },
            Some((_, Node::TraitItem(i))) => {
                if let TraitItemKind::Fn(sig, _) = &i.kind {
                    (i.owner_id, sig, true)
                } else {
                    return;
                }
            },
            _ => return,
        };

        check_mut_from_ref(cx, sig, Some(body));

        if !matches!(sig.header.abi, Abi::Rust) {
            // Ignore `extern` functions with non-Rust calling conventions
            return;
        }

        let decl = sig.decl;
        let sig = cx.tcx.fn_sig(item_id).instantiate_identity().skip_binder();
        let lint_args: Vec<_> = check_fn_args(cx, sig, decl.inputs, body.params)
            .filter(|arg| !is_trait_item || arg.mutability() == Mutability::Not)
            .collect();
        let results = check_ptr_arg_usage(cx, body, &lint_args);

        for (result, args) in results.iter().zip(lint_args.iter()).filter(|(r, _)| !r.skip) {
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

    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, l, r) = expr.kind {
            if (op.node == BinOpKind::Eq || op.node == BinOpKind::Ne) && (is_null_path(cx, l) || is_null_path(cx, r)) {
                span_lint(
                    cx,
                    CMP_NULL,
                    expr.span,
                    "comparing with null is better expressed by the `.is_null()` method",
                );
            }
        } else {
            check_invalid_ptr_usage(cx, expr);
        }
    }
}

fn check_invalid_ptr_usage<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
    if let ExprKind::Call(fun, args) = expr.kind
        && let ExprKind::Path(ref qpath) = fun.kind
        && let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id()
        && let Some(name) = cx.tcx.get_diagnostic_name(fun_def_id)
    {
        // TODO: `ptr_slice_from_raw_parts` and its mutable variant should probably still be linted
        // conditionally based on how the return value is used, but not universally like the other
        // functions since there are valid uses for null slice pointers.
        //
        // See: https://github.com/rust-lang/rust-clippy/pull/13452/files#r1773772034

        // `arg` positions where null would cause U.B.
        let arg_indices: &[_] = match name {
            sym::ptr_read
            | sym::ptr_read_unaligned
            | sym::ptr_read_volatile
            | sym::ptr_replace
            | sym::ptr_write
            | sym::ptr_write_bytes
            | sym::ptr_write_unaligned
            | sym::ptr_write_volatile
            | sym::slice_from_raw_parts
            | sym::slice_from_raw_parts_mut => &[0],
            sym::ptr_copy | sym::ptr_copy_nonoverlapping | sym::ptr_swap | sym::ptr_swap_nonoverlapping => &[0, 1],
            _ => return,
        };

        for &arg_idx in arg_indices {
            if let Some(arg) = args.get(arg_idx).filter(|arg| is_null_path(cx, arg))
                && let Some(std_or_core) = std_or_core(cx)
            {
                span_lint_and_sugg(
                    cx,
                    INVALID_NULL_PTR_USAGE,
                    arg.span,
                    "pointer must be non-null",
                    "change this to",
                    format!("{std_or_core}::ptr::NonNull::dangling().as_ptr()"),
                    Applicability::MachineApplicable,
                );
            }
        }
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
    method_renames: &'static [(&'static str, &'static str)],
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
    fn_sig
        .inputs()
        .iter()
        .zip(hir_tys.iter())
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
                        [("clone", ".to_owned()")].as_slice(),
                        DerefTy::Slice(
                            name.args.and_then(|args| args.args.first()).and_then(|arg| {
                                if let GenericArg::Type(ty) = arg {
                                    Some(ty.span)
                                } else {
                                    None
                                }
                            }),
                            args.type_at(0),
                        ),
                    ),
                    _ if Some(adt.did()) == cx.tcx.lang_items().string() => {
                        ([("clone", ".to_owned()"), ("as_str", "")].as_slice(), DerefTy::Str)
                    },
                    Some(sym::PathBuf) => ([("clone", ".to_path_buf()"), ("as_path", "")].as_slice(), DerefTy::Path),
                    Some(sym::Cow) if mutability == Mutability::Not => {
                        if let Some((lifetime, ty)) = name.args.and_then(|args| {
                            if let [GenericArg::Lifetime(lifetime), ty] = args.args {
                                return Some((lifetime, ty));
                            }
                            None
                        }) {
                            if let LifetimeName::Param(param_def_id) = lifetime.res
                                && !lifetime.is_anonymous()
                                && fn_sig
                                    .output()
                                    .walk()
                                    .filter_map(|arg| {
                                        arg.as_region().and_then(|lifetime| match lifetime.kind() {
                                            ty::ReEarlyParam(r) => Some(
                                                cx.tcx
                                                    .generics_of(cx.tcx.parent(param_def_id.to_def_id()))
                                                    .region_param(r, cx.tcx)
                                                    .def_id,
                                            ),
                                            ty::ReBound(_, r) => r.kind.get_id(),
                                            ty::ReLateParam(r) => r.bound_region.get_id(),
                                            ty::ReStatic
                                            | ty::ReVar(_)
                                            | ty::RePlaceholder(_)
                                            | ty::ReErased
                                            | ty::ReError(_) => None,
                                        })
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

fn check_mut_from_ref<'tcx>(cx: &LateContext<'tcx>, sig: &FnSig<'_>, body: Option<&Body<'tcx>>) {
    if let FnRetTy::Return(ty) = sig.decl.output
        && let Some((out, Mutability::Mut, _)) = get_ref_lm(ty)
    {
        let out_region = cx.tcx.named_bound_var(out.hir_id);
        let args: Option<Vec<_>> = sig
            .decl
            .inputs
            .iter()
            .filter_map(get_ref_lm)
            .filter(|&(lt, _, _)| cx.tcx.named_bound_var(lt.hir_id) == out_region)
            .map(|(_, mutability, span)| (mutability == Mutability::Not).then_some(span))
            .collect();
        if let Some(args) = args
            && !args.is_empty()
            && body.is_none_or(|body| sig.header.safety == Safety::Unsafe || contains_unsafe_block(cx, body.value))
        {
            span_lint_and_then(
                cx,
                MUT_FROM_REF,
                ty.span,
                "mutable borrow from immutable input(s)",
                |diag| {
                    let ms = MultiSpan::from_spans(args);
                    diag.span_note(ms, "immutable borrow here");
                },
            );
        }
    }
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
        fn nested_visit_map(&mut self) -> Self::Map {
            self.cx.tcx.hir()
        }

        fn visit_anon_const(&mut self, _: &'tcx AnonConst) {}

        fn visit_expr(&mut self, e: &'tcx Expr<'_>) {
            if self.skip_count == self.args.len() {
                return;
            }

            // Check if this is local we care about
            let Some(&args_idx) = path_to_local(e).and_then(|id| self.bindings.get(&id)) else {
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
                    if let PatKind::Binding(BindingMode::NONE, id, _, None) = l.pat.kind {
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
                        let name = name.ident.as_str();

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
                        if is_allowed_vec_method(self.cx, use_expr) {
                            return;
                        }
                    }

                    let deref_ty = args.deref_ty.ty(self.cx);
                    let adjusted_ty = self.cx.typeck_results().expr_ty_adjusted(e).peel_refs();
                    if adjusted_ty == deref_ty {
                        return;
                    }

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
                    PatKind::Binding(BindingMode::NONE, id, _, None) if !is_lint_allowed(cx, PTR_ARG, param.hir_id) => {
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

fn get_ref_lm<'tcx>(ty: &'tcx hir::Ty<'tcx>) -> Option<(&'tcx Lifetime, Mutability, Span)> {
    if let TyKind::Ref(lt, ref m) = ty.kind {
        Some((lt, m.mutbl, ty.span))
    } else {
        None
    }
}

fn is_null_path(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    if let ExprKind::Call(pathexp, []) = expr.kind {
        path_def_id(cx, pathexp)
            .is_some_and(|id| matches!(cx.tcx.get_diagnostic_name(id), Some(sym::ptr_null | sym::ptr_null_mut)))
    } else {
        false
    }
}
