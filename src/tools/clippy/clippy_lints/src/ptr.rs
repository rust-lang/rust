//! Checks for usage of  `&Vec[_]` and `&String`.

use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg, span_lint_and_then, span_lint_hir_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::expr_sig;
use clippy_utils::visitors::contains_unsafe_block;
use clippy_utils::{get_expr_use_or_unification_node, is_lint_allowed, path_def_id, path_to_local, paths};
use if_chain::if_chain;
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::def_id::DefId;
use rustc_hir::hir_id::HirIdMap;
use rustc_hir::intravisit::{walk_expr, Visitor};
use rustc_hir::{
    self as hir, AnonConst, BinOpKind, BindingAnnotation, Body, Expr, ExprKind, FnRetTy, FnSig, GenericArg,
    ImplItemKind, ItemKind, Lifetime, Mutability, Node, Param, PatKind, QPath, TraitFn, TraitItem, TraitItemKind,
    TyKind, Unsafety,
};
use rustc_infer::infer::TyCtxtInferExt;
use rustc_infer::traits::{Obligation, ObligationCause};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, Binder, Clause, ExistentialPredicate, List, PredicateKind, Ty};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;
use rustc_span::sym;
use rustc_span::symbol::Symbol;
use rustc_trait_selection::infer::InferCtxtExt as _;
use rustc_trait_selection::traits::query::evaluate_obligation::InferCtxtExt as _;
use std::fmt;
use std::iter;

declare_clippy_lint! {
    /// ### What it does
    /// This lint checks for function arguments of type `&String`, `&Vec`,
    /// `&PathBuf`, and `Cow<_>`. It will also suggest you replace `.clone()` calls
    /// with the appropriate `.to_owned()`/`to_string()` calls.
    ///
    /// ### Why is this bad?
    /// Requiring the argument to be of the specific size
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
            for arg in check_fn_args(
                cx,
                cx.tcx.fn_sig(item.owner_id).subst_identity().skip_binder().inputs(),
                sig.decl.inputs,
                &[],
            )
            .filter(|arg| arg.mutability() == Mutability::Not)
            {
                span_lint_hir_and_then(cx, PTR_ARG, arg.emission_id, arg.span, &arg.build_msg(), |diag| {
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

    fn check_body(&mut self, cx: &LateContext<'tcx>, body: &'tcx Body<'_>) {
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
        let decl = sig.decl;
        let sig = cx.tcx.fn_sig(item_id).subst_identity().skip_binder();
        let lint_args: Vec<_> = check_fn_args(cx, sig.inputs(), decl.inputs, body.params)
            .filter(|arg| !is_trait_item || arg.mutability() == Mutability::Not)
            .collect();
        let results = check_ptr_arg_usage(cx, body, &lint_args);

        for (result, args) in results.iter().zip(lint_args.iter()).filter(|(r, _)| !r.skip) {
            span_lint_hir_and_then(cx, PTR_ARG, args.emission_id, args.span, &args.build_msg(), |diag| {
                diag.multipart_suggestion(
                    "change this to",
                    iter::once((args.span, format!("{}{}", args.ref_prefix, args.deref_ty.display(cx))))
                        .chain(result.replacements.iter().map(|r| {
                            (
                                r.expr_span,
                                format!("{}{}", snippet_opt(cx, r.self_span).unwrap(), r.replacement),
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
    // (fn_path, arg_indices) - `arg_indices` are the `arg` positions where null would cause U.B.
    const INVALID_NULL_PTR_USAGE_TABLE: [(&[&str], &[usize]); 16] = [
        (&paths::SLICE_FROM_RAW_PARTS, &[0]),
        (&paths::SLICE_FROM_RAW_PARTS_MUT, &[0]),
        (&paths::PTR_COPY, &[0, 1]),
        (&paths::PTR_COPY_NONOVERLAPPING, &[0, 1]),
        (&paths::PTR_READ, &[0]),
        (&paths::PTR_READ_UNALIGNED, &[0]),
        (&paths::PTR_READ_VOLATILE, &[0]),
        (&paths::PTR_REPLACE, &[0]),
        (&paths::PTR_SLICE_FROM_RAW_PARTS, &[0]),
        (&paths::PTR_SLICE_FROM_RAW_PARTS_MUT, &[0]),
        (&paths::PTR_SWAP, &[0, 1]),
        (&paths::PTR_SWAP_NONOVERLAPPING, &[0, 1]),
        (&paths::PTR_WRITE, &[0]),
        (&paths::PTR_WRITE_UNALIGNED, &[0]),
        (&paths::PTR_WRITE_VOLATILE, &[0]),
        (&paths::PTR_WRITE_BYTES, &[0]),
    ];

    if_chain! {
        if let ExprKind::Call(fun, args) = expr.kind;
        if let ExprKind::Path(ref qpath) = fun.kind;
        if let Some(fun_def_id) = cx.qpath_res(qpath, fun.hir_id).opt_def_id();
        let fun_def_path = cx.get_def_path(fun_def_id).into_iter().map(Symbol::to_ident_string).collect::<Vec<_>>();
        if let Some(&(_, arg_indices)) = INVALID_NULL_PTR_USAGE_TABLE
            .iter()
            .find(|&&(fn_path, _)| fn_path == fun_def_path);
        then {
            for &arg_idx in arg_indices {
                if let Some(arg) = args.get(arg_idx).filter(|arg| is_null_path(cx, arg)) {
                    span_lint_and_sugg(
                        cx,
                        INVALID_NULL_PTR_USAGE,
                        arg.span,
                        "pointer must be non-null",
                        "change this to",
                        "core::ptr::NonNull::dangling().as_ptr()".to_string(),
                        Applicability::MachineApplicable,
                    );
                }
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
    emission_id: hir::HirId,
    span: Span,
    ty_did: DefId,
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
                match hir_ty.and_then(|s| snippet_opt(self.0, s)) {
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
            Self::Path => cx.tcx.mk_adt(
                cx.tcx.adt_def(cx.tcx.get_diagnostic_item(sym::Path).unwrap()),
                List::empty(),
            ),
            Self::Slice(_, ty) => cx.tcx.mk_slice(ty),
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
    tys: &'tcx [Ty<'tcx>],
    hir_tys: &'tcx [hir::Ty<'tcx>],
    params: &'tcx [Param<'tcx>],
) -> impl Iterator<Item = PtrArg<'tcx>> + 'cx {
    tys.iter()
        .zip(hir_tys.iter())
        .enumerate()
        .filter_map(|(i, (ty, hir_ty))| {
            if_chain! {
                if let ty::Ref(_, ty, mutability) = *ty.kind();
                if let ty::Adt(adt, substs) = *ty.kind();

                if let TyKind::Ref(lt, ref ty) = hir_ty.kind;
                if let TyKind::Path(QPath::Resolved(None, path)) = ty.ty.kind;

                // Check that the name as typed matches the actual name of the type.
                // e.g. `fn foo(_: &Foo)` shouldn't trigger the lint when `Foo` is an alias for `Vec`
                if let [.., name] = path.segments;
                if cx.tcx.item_name(adt.did()) == name.ident.name;

                then {
                    let emission_id = params.get(i).map_or(hir_ty.hir_id, |param| param.hir_id);
                    let (method_renames, deref_ty) = match cx.tcx.get_diagnostic_name(adt.did()) {
                        Some(sym::Vec) => (
                            [("clone", ".to_owned()")].as_slice(),
                            DerefTy::Slice(
                                name.args
                                    .and_then(|args| args.args.first())
                                    .and_then(|arg| if let GenericArg::Type(ty) = arg {
                                        Some(ty.span)
                                    } else {
                                        None
                                    }),
                                substs.type_at(0),
                            ),
                        ),
                        _ if Some(adt.did()) == cx.tcx.lang_items().string() => (
                            [("clone", ".to_owned()"), ("as_str", "")].as_slice(),
                            DerefTy::Str,
                        ),
                        Some(sym::PathBuf) => (
                            [("clone", ".to_path_buf()"), ("as_path", "")].as_slice(),
                            DerefTy::Path,
                        ),
                        Some(sym::Cow) if mutability == Mutability::Not => {
                            let ty_name = name.args
                                .and_then(|args| {
                                    args.args.iter().find_map(|a| match a {
                                        GenericArg::Type(x) => Some(x),
                                        _ => None,
                                    })
                                })
                                .and_then(|arg| snippet_opt(cx, arg.span))
                                .unwrap_or_else(|| substs.type_at(1).to_string());
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
                                        format!("&{}{ty_name}", mutability.prefix_str()),
                                        Applicability::Unspecified,
                                    );
                                }
                            );
                            return None;
                        },
                        _ => return None,
                    };
                    return Some(PtrArg {
                        idx: i,
                        emission_id,
                        span: hir_ty.span,
                        ty_did: adt.did(),
                        ty_name: name.ident.name,
                        method_renames,
                        ref_prefix: RefPrefix {
                            lt: *lt,
                            mutability,
                        },
                        deref_ty,
                    });
                }
            }
            None
        })
}

fn check_mut_from_ref<'tcx>(cx: &LateContext<'tcx>, sig: &FnSig<'_>, body: Option<&'tcx Body<'_>>) {
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
            && body.map_or(true, |body| {
                sig.header.unsafety == Unsafety::Unsafe || contains_unsafe_block(cx, body.value)
            })
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
fn check_ptr_arg_usage<'tcx>(cx: &LateContext<'tcx>, body: &'tcx Body<'_>, args: &[PtrArg<'tcx>]) -> Vec<PtrArgResult> {
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
                Some((Node::Local(l), _)) => {
                    // Only trace simple bindings. e.g `let x = y;`
                    if let PatKind::Binding(BindingAnnotation::NONE, id, _, None) = l.pat.kind {
                        self.bindings.insert(id, args_idx);
                    } else {
                        set_skip_flag();
                    }
                },
                Some((Node::Expr(e), child_id)) => match e.kind {
                    ExprKind::Call(f, expr_args) => {
                        let i = expr_args.iter().position(|arg| arg.hir_id == child_id).unwrap_or(0);
                        if expr_sig(self.cx, f).and_then(|sig| sig.input(i)).map_or(true, |ty| {
                            match *ty.skip_binder().peel_refs().kind() {
                                ty::Dynamic(preds, _, _) => !matches_preds(self.cx, args.deref_ty.ty(self.cx), preds),
                                ty::Param(_) => true,
                                ty::Adt(def, _) => def.did() == args.ty_did,
                                _ => false,
                            }
                        }) {
                            // Passed to a function taking the non-dereferenced type.
                            set_skip_flag();
                        }
                    },
                    ExprKind::MethodCall(name, self_arg, expr_args, _) => {
                        let i = std::iter::once(self_arg)
                            .chain(expr_args.iter())
                            .position(|arg| arg.hir_id == child_id)
                            .unwrap_or(0);
                        if i == 0 {
                            // Check if the method can be renamed.
                            let name = name.ident.as_str();
                            if let Some((_, replacement)) = args.method_renames.iter().find(|&&(x, _)| x == name) {
                                result.replacements.push(PtrArgReplacement {
                                    expr_span: e.span,
                                    self_span: self_arg.span,
                                    replacement,
                                });
                                return;
                            }
                        }

                        let Some(id) = self.cx.typeck_results().type_dependent_def_id(e.hir_id) else {
                            set_skip_flag();
                            return;
                        };

                        match *self.cx.tcx.fn_sig(id).subst_identity().skip_binder().inputs()[i]
                            .peel_refs()
                            .kind()
                        {
                            ty::Dynamic(preds, _, _) if !matches_preds(self.cx, args.deref_ty.ty(self.cx), preds) => {
                                set_skip_flag();
                            },
                            ty::Param(_) => {
                                set_skip_flag();
                            },
                            // If the types match check for methods which exist on both types. e.g. `Vec::len` and
                            // `slice::len`
                            ty::Adt(def, _) if def.did() == args.ty_did => {
                                set_skip_flag();
                            },
                            _ => (),
                        }
                    },
                    // Indexing is fine for currently supported types.
                    ExprKind::Index(e, _) if e.hir_id == child_id => (),
                    _ => set_skip_flag(),
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
                    PatKind::Binding(BindingAnnotation::NONE, id, _, None)
                        if !is_lint_allowed(cx, PTR_ARG, param.hir_id) =>
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
    let infcx = cx.tcx.infer_ctxt().build();
    preds.iter().all(|&p| match cx.tcx.erase_late_bound_regions(p) {
        ExistentialPredicate::Trait(p) => infcx
            .type_implements_trait(p.def_id, [ty.into()].into_iter().chain(p.substs.iter()), cx.param_env)
            .must_apply_modulo_regions(),
        ExistentialPredicate::Projection(p) => infcx.predicate_must_hold_modulo_regions(&Obligation::new(
            cx.tcx,
            ObligationCause::dummy(),
            cx.param_env,
            cx.tcx
                .mk_predicate(Binder::dummy(PredicateKind::Clause(Clause::Projection(
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
        path_def_id(cx, pathexp).map_or(false, |id| {
            matches!(cx.tcx.get_diagnostic_name(id), Some(sym::ptr_null | sym::ptr_null_mut))
        })
    } else {
        false
    }
}
