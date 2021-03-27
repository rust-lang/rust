mod not_unsafe_ptr_arg_deref;
mod too_many_arguments;
mod too_many_lines;

use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_then};
use clippy_utils::source::snippet_opt;
use clippy_utils::ty::{is_must_use_ty, is_type_diagnostic_item};
use clippy_utils::{attr_by_name, attrs::is_proc_macro, match_def_path, must_use_attr, return_ty, trait_ref_of_method};
use if_chain::if_chain;
use rustc_ast::ast::Attribute;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::intravisit;
use rustc_hir::{def::Res, def_id::DefId, QPath};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::hir::map::Map;
use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Ty};
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::{sym, Span};
use rustc_typeck::hir_ty_to_ty;

declare_clippy_lint! {
    /// **What it does:** Checks for functions with too many parameters.
    ///
    /// **Why is this bad?** Functions with lots of parameters are considered bad
    /// style and reduce readability (“what does the 5th parameter mean?”). Consider
    /// grouping some parameters into a new type.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// # struct Color;
    /// fn foo(x: u32, y: u32, name: &str, c: Color, w: f32, h: f32, a: f32, b: f32) {
    ///     // ..
    /// }
    /// ```
    pub TOO_MANY_ARGUMENTS,
    complexity,
    "functions with too many arguments"
}

declare_clippy_lint! {
    /// **What it does:** Checks for functions with a large amount of lines.
    ///
    /// **Why is this bad?** Functions with a lot of lines are harder to understand
    /// due to having to look at a larger amount of code to understand what the
    /// function is doing. Consider splitting the body of the function into
    /// multiple functions.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust
    /// fn im_too_long() {
    ///     println!("");
    ///     // ... 100 more LoC
    ///     println!("");
    /// }
    /// ```
    pub TOO_MANY_LINES,
    pedantic,
    "functions with too many lines"
}

declare_clippy_lint! {
    /// **What it does:** Checks for public functions that dereference raw pointer
    /// arguments but are not marked unsafe.
    ///
    /// **Why is this bad?** The function should probably be marked `unsafe`, since
    /// for an arbitrary raw pointer, there is no way of telling for sure if it is
    /// valid.
    ///
    /// **Known problems:**
    ///
    /// * It does not check functions recursively so if the pointer is passed to a
    /// private non-`unsafe` function which does the dereferencing, the lint won't
    /// trigger.
    /// * It only checks for arguments whose type are raw pointers, not raw pointers
    /// got from an argument in some other way (`fn foo(bar: &[*const u8])` or
    /// `some_argument.get_raw_ptr()`).
    ///
    /// **Example:**
    /// ```rust,ignore
    /// // Bad
    /// pub fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    ///
    /// // Good
    /// pub unsafe fn foo(x: *const u8) {
    ///     println!("{}", unsafe { *x });
    /// }
    /// ```
    pub NOT_UNSAFE_PTR_ARG_DEREF,
    correctness,
    "public functions dereferencing raw pointer arguments but not marked `unsafe`"
}

declare_clippy_lint! {
    /// **What it does:** Checks for a [`#[must_use]`] attribute on
    /// unit-returning functions and methods.
    ///
    /// [`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
    ///
    /// **Why is this bad?** Unit values are useless. The attribute is likely
    /// a remnant of a refactoring that removed the return type.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
    /// ```rust
    /// #[must_use]
    /// fn useless() { }
    /// ```
    pub MUST_USE_UNIT,
    style,
    "`#[must_use]` attribute on a unit-returning function / method"
}

declare_clippy_lint! {
    /// **What it does:** Checks for a [`#[must_use]`] attribute without
    /// further information on functions and methods that return a type already
    /// marked as `#[must_use]`.
    ///
    /// [`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
    ///
    /// **Why is this bad?** The attribute isn't needed. Not using the result
    /// will already be reported. Alternatively, one can add some text to the
    /// attribute to improve the lint message.
    ///
    /// **Known problems:** None.
    ///
    /// **Examples:**
    /// ```rust
    /// #[must_use]
    /// fn double_must_use() -> Result<(), ()> {
    ///     unimplemented!();
    /// }
    /// ```
    pub DOUBLE_MUST_USE,
    style,
    "`#[must_use]` attribute on a `#[must_use]`-returning function / method"
}

declare_clippy_lint! {
    /// **What it does:** Checks for public functions that have no
    /// [`#[must_use]`] attribute, but return something not already marked
    /// must-use, have no mutable arg and mutate no statics.
    ///
    /// [`#[must_use]`]: https://doc.rust-lang.org/reference/attributes/diagnostics.html#the-must_use-attribute
    ///
    /// **Why is this bad?** Not bad at all, this lint just shows places where
    /// you could add the attribute.
    ///
    /// **Known problems:** The lint only checks the arguments for mutable
    /// types without looking if they are actually changed. On the other hand,
    /// it also ignores a broad range of potentially interesting side effects,
    /// because we cannot decide whether the programmer intends the function to
    /// be called for the side effect or the result. Expect many false
    /// positives. At least we don't lint if the result type is unit or already
    /// `#[must_use]`.
    ///
    /// **Examples:**
    /// ```rust
    /// // this could be annotated with `#[must_use]`.
    /// fn id<T>(t: T) -> T { t }
    /// ```
    pub MUST_USE_CANDIDATE,
    pedantic,
    "function or method that could take a `#[must_use]` attribute"
}

declare_clippy_lint! {
    /// **What it does:** Checks for public functions that return a `Result`
    /// with an `Err` type of `()`. It suggests using a custom type that
    /// implements [`std::error::Error`].
    ///
    /// **Why is this bad?** Unit does not implement `Error` and carries no
    /// further information about what went wrong.
    ///
    /// **Known problems:** Of course, this lint assumes that `Result` is used
    /// for a fallible operation (which is after all the intended use). However
    /// code may opt to (mis)use it as a basic two-variant-enum. In that case,
    /// the suggestion is misguided, and the code should use a custom enum
    /// instead.
    ///
    /// **Examples:**
    /// ```rust
    /// pub fn read_u8() -> Result<u8, ()> { Err(()) }
    /// ```
    /// should become
    /// ```rust,should_panic
    /// use std::fmt;
    ///
    /// #[derive(Debug)]
    /// pub struct EndOfStream;
    ///
    /// impl fmt::Display for EndOfStream {
    ///     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    ///         write!(f, "End of Stream")
    ///     }
    /// }
    ///
    /// impl std::error::Error for EndOfStream { }
    ///
    /// pub fn read_u8() -> Result<u8, EndOfStream> { Err(EndOfStream) }
    ///# fn main() {
    ///#     read_u8().unwrap();
    ///# }
    /// ```
    ///
    /// Note that there are crates that simplify creating the error type, e.g.
    /// [`thiserror`](https://docs.rs/thiserror).
    pub RESULT_UNIT_ERR,
    style,
    "public function returning `Result` with an `Err` type of `()`"
}

#[derive(Copy, Clone)]
pub struct Functions {
    too_many_arguments_threshold: u64,
    too_many_lines_threshold: u64,
}

impl Functions {
    pub fn new(too_many_arguments_threshold: u64, too_many_lines_threshold: u64) -> Self {
        Self {
            too_many_arguments_threshold,
            too_many_lines_threshold,
        }
    }
}

impl_lint_pass!(Functions => [
    TOO_MANY_ARGUMENTS,
    TOO_MANY_LINES,
    NOT_UNSAFE_PTR_ARG_DEREF,
    MUST_USE_UNIT,
    DOUBLE_MUST_USE,
    MUST_USE_CANDIDATE,
    RESULT_UNIT_ERR,
]);

impl<'tcx> LateLintPass<'tcx> for Functions {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        kind: intravisit::FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'_>,
        body: &'tcx hir::Body<'_>,
        span: Span,
        hir_id: hir::HirId,
    ) {
        too_many_arguments::check_fn(cx, kind, decl, span, hir_id, self.too_many_arguments_threshold);
        too_many_lines::check(cx, span, body, self.too_many_lines_threshold);
        not_unsafe_ptr_arg_deref::check_fn(cx, kind, decl, body, hir_id);
    }

    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::Item<'_>) {
        let attrs = cx.tcx.hir().attrs(item.hir_id());
        let attr = must_use_attr(attrs);
        if let hir::ItemKind::Fn(ref sig, ref _generics, ref body_id) = item.kind {
            let is_public = cx.access_levels.is_exported(item.hir_id());
            let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
            if is_public {
                check_result_unit_err(cx, &sig.decl, item.span, fn_header_span);
            }
            if let Some(attr) = attr {
                check_needless_must_use(cx, &sig.decl, item.hir_id(), item.span, fn_header_span, attr);
                return;
            }
            if is_public && !is_proc_macro(cx.sess(), attrs) && attr_by_name(attrs, "no_mangle").is_none() {
                check_must_use_candidate(
                    cx,
                    &sig.decl,
                    cx.tcx.hir().body(*body_id),
                    item.span,
                    item.hir_id(),
                    item.span.with_hi(sig.decl.output.span().hi()),
                    "this function could have a `#[must_use]` attribute",
                );
            }
        }
    }

    fn check_impl_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ImplItem<'_>) {
        if let hir::ImplItemKind::Fn(ref sig, ref body_id) = item.kind {
            let is_public = cx.access_levels.is_exported(item.hir_id());
            let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
            if is_public && trait_ref_of_method(cx, item.hir_id()).is_none() {
                check_result_unit_err(cx, &sig.decl, item.span, fn_header_span);
            }
            let attrs = cx.tcx.hir().attrs(item.hir_id());
            let attr = must_use_attr(attrs);
            if let Some(attr) = attr {
                check_needless_must_use(cx, &sig.decl, item.hir_id(), item.span, fn_header_span, attr);
            } else if is_public && !is_proc_macro(cx.sess(), attrs) && trait_ref_of_method(cx, item.hir_id()).is_none()
            {
                check_must_use_candidate(
                    cx,
                    &sig.decl,
                    cx.tcx.hir().body(*body_id),
                    item.span,
                    item.hir_id(),
                    item.span.with_hi(sig.decl.output.span().hi()),
                    "this method could have a `#[must_use]` attribute",
                );
            }
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'_>) {
        too_many_arguments::check_trait_item(cx, item, self.too_many_arguments_threshold);
        not_unsafe_ptr_arg_deref::check_trait_item(cx, item);

        if let hir::TraitItemKind::Fn(ref sig, ref eid) = item.kind {
            let is_public = cx.access_levels.is_exported(item.hir_id());
            let fn_header_span = item.span.with_hi(sig.decl.output.span().hi());
            if is_public {
                check_result_unit_err(cx, &sig.decl, item.span, fn_header_span);
            }

            let attrs = cx.tcx.hir().attrs(item.hir_id());
            let attr = must_use_attr(attrs);
            if let Some(attr) = attr {
                check_needless_must_use(cx, &sig.decl, item.hir_id(), item.span, fn_header_span, attr);
            }
            if let hir::TraitFn::Provided(eid) = *eid {
                let body = cx.tcx.hir().body(eid);
                if attr.is_none() && is_public && !is_proc_macro(cx.sess(), attrs) {
                    check_must_use_candidate(
                        cx,
                        &sig.decl,
                        body,
                        item.span,
                        item.hir_id(),
                        item.span.with_hi(sig.decl.output.span().hi()),
                        "this method could have a `#[must_use]` attribute",
                    );
                }
            }
        }
    }
}

fn check_result_unit_err(cx: &LateContext<'_>, decl: &hir::FnDecl<'_>, item_span: Span, fn_header_span: Span) {
    if_chain! {
        if !in_external_macro(cx.sess(), item_span);
        if let hir::FnRetTy::Return(ref ty) = decl.output;
        let ty = hir_ty_to_ty(cx.tcx, ty);
        if is_type_diagnostic_item(cx, ty, sym::result_type);
        if let ty::Adt(_, substs) = ty.kind();
        let err_ty = substs.type_at(1);
        if err_ty.is_unit();
        then {
            span_lint_and_help(
                cx,
                RESULT_UNIT_ERR,
                fn_header_span,
                "this returns a `Result<_, ()>",
                None,
                "use a custom Error type instead",
            );
        }
    }
}

fn check_needless_must_use(
    cx: &LateContext<'_>,
    decl: &hir::FnDecl<'_>,
    item_id: hir::HirId,
    item_span: Span,
    fn_header_span: Span,
    attr: &Attribute,
) {
    if in_external_macro(cx.sess(), item_span) {
        return;
    }
    if returns_unit(decl) {
        span_lint_and_then(
            cx,
            MUST_USE_UNIT,
            fn_header_span,
            "this unit-returning function has a `#[must_use]` attribute",
            |diag| {
                diag.span_suggestion(
                    attr.span,
                    "remove the attribute",
                    "".into(),
                    Applicability::MachineApplicable,
                );
            },
        );
    } else if !attr.is_value_str() && is_must_use_ty(cx, return_ty(cx, item_id)) {
        span_lint_and_help(
            cx,
            DOUBLE_MUST_USE,
            fn_header_span,
            "this function has an empty `#[must_use]` attribute, but returns a type already marked as `#[must_use]`",
            None,
            "either add some descriptive text or remove the attribute",
        );
    }
}

fn check_must_use_candidate<'tcx>(
    cx: &LateContext<'tcx>,
    decl: &'tcx hir::FnDecl<'_>,
    body: &'tcx hir::Body<'_>,
    item_span: Span,
    item_id: hir::HirId,
    fn_span: Span,
    msg: &str,
) {
    if has_mutable_arg(cx, body)
        || mutates_static(cx, body)
        || in_external_macro(cx.sess(), item_span)
        || returns_unit(decl)
        || !cx.access_levels.is_exported(item_id)
        || is_must_use_ty(cx, return_ty(cx, item_id))
    {
        return;
    }
    span_lint_and_then(cx, MUST_USE_CANDIDATE, fn_span, msg, |diag| {
        if let Some(snippet) = snippet_opt(cx, fn_span) {
            diag.span_suggestion(
                fn_span,
                "add the attribute",
                format!("#[must_use] {}", snippet),
                Applicability::MachineApplicable,
            );
        }
    });
}

fn returns_unit(decl: &hir::FnDecl<'_>) -> bool {
    match decl.output {
        hir::FnRetTy::DefaultReturn(_) => true,
        hir::FnRetTy::Return(ref ty) => match ty.kind {
            hir::TyKind::Tup(ref tys) => tys.is_empty(),
            hir::TyKind::Never => true,
            _ => false,
        },
    }
}

fn has_mutable_arg(cx: &LateContext<'_>, body: &hir::Body<'_>) -> bool {
    let mut tys = FxHashSet::default();
    body.params.iter().any(|param| is_mutable_pat(cx, &param.pat, &mut tys))
}

fn is_mutable_pat(cx: &LateContext<'_>, pat: &hir::Pat<'_>, tys: &mut FxHashSet<DefId>) -> bool {
    if let hir::PatKind::Wild = pat.kind {
        return false; // ignore `_` patterns
    }
    if cx.tcx.has_typeck_results(pat.hir_id.owner.to_def_id()) {
        is_mutable_ty(cx, &cx.tcx.typeck(pat.hir_id.owner).pat_ty(pat), pat.span, tys)
    } else {
        false
    }
}

static KNOWN_WRAPPER_TYS: &[&[&str]] = &[&["alloc", "rc", "Rc"], &["std", "sync", "Arc"]];

fn is_mutable_ty<'tcx>(cx: &LateContext<'tcx>, ty: Ty<'tcx>, span: Span, tys: &mut FxHashSet<DefId>) -> bool {
    match *ty.kind() {
        // primitive types are never mutable
        ty::Bool | ty::Char | ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Str => false,
        ty::Adt(ref adt, ref substs) => {
            tys.insert(adt.did) && !ty.is_freeze(cx.tcx.at(span), cx.param_env)
                || KNOWN_WRAPPER_TYS.iter().any(|path| match_def_path(cx, adt.did, path))
                    && substs.types().any(|ty| is_mutable_ty(cx, ty, span, tys))
        },
        ty::Tuple(ref substs) => substs.types().any(|ty| is_mutable_ty(cx, ty, span, tys)),
        ty::Array(ty, _) | ty::Slice(ty) => is_mutable_ty(cx, ty, span, tys),
        ty::RawPtr(ty::TypeAndMut { ty, mutbl }) | ty::Ref(_, ty, mutbl) => {
            mutbl == hir::Mutability::Mut || is_mutable_ty(cx, ty, span, tys)
        },
        // calling something constitutes a side effect, so return true on all callables
        // also never calls need not be used, so return true for them, too
        _ => true,
    }
}

struct StaticMutVisitor<'a, 'tcx> {
    cx: &'a LateContext<'tcx>,
    mutates_static: bool,
}

impl<'a, 'tcx> intravisit::Visitor<'tcx> for StaticMutVisitor<'a, 'tcx> {
    type Map = Map<'tcx>;

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'_>) {
        use hir::ExprKind::{AddrOf, Assign, AssignOp, Call, MethodCall};

        if self.mutates_static {
            return;
        }
        match expr.kind {
            Call(_, args) | MethodCall(_, _, args, _) => {
                let mut tys = FxHashSet::default();
                for arg in args {
                    if self.cx.tcx.has_typeck_results(arg.hir_id.owner.to_def_id())
                        && is_mutable_ty(
                            self.cx,
                            self.cx.tcx.typeck(arg.hir_id.owner).expr_ty(arg),
                            arg.span,
                            &mut tys,
                        )
                        && is_mutated_static(arg)
                    {
                        self.mutates_static = true;
                        return;
                    }
                    tys.clear();
                }
            },
            Assign(ref target, ..) | AssignOp(_, ref target, _) | AddrOf(_, hir::Mutability::Mut, ref target) => {
                self.mutates_static |= is_mutated_static(target)
            },
            _ => {},
        }
    }

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::None
    }
}

fn is_mutated_static(e: &hir::Expr<'_>) -> bool {
    use hir::ExprKind::{Field, Index, Path};

    match e.kind {
        Path(QPath::Resolved(_, path)) => !matches!(path.res, Res::Local(_)),
        Path(_) => true,
        Field(ref inner, _) | Index(ref inner, _) => is_mutated_static(inner),
        _ => false,
    }
}

fn mutates_static<'tcx>(cx: &LateContext<'tcx>, body: &'tcx hir::Body<'_>) -> bool {
    let mut v = StaticMutVisitor {
        cx,
        mutates_static: false,
    };
    intravisit::walk_expr(&mut v, &body.value);
    v.mutates_static
}
