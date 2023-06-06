use clippy_utils::diagnostics::span_lint;
use rustc_ast::LitKind;
use rustc_hir as hir;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty::Ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Warns about needless / redundant type annotations.
    ///
    /// ### Why is this bad?
    /// Code without type annotations is shorter and in most cases
    /// more idiomatic and easier to modify.
    ///
    /// ### Limitations
    /// This lint doesn't support:
    ///
    /// - Generics
    /// - Refs returned from anything else than a `MethodCall`
    /// - Complex types (tuples, arrays, etc...)
    /// - `Path` to anything else than a primitive type.
    ///
    /// ### Example
    /// ```rust
    /// let foo: String = String::new();
    /// ```
    /// Use instead:
    /// ```rust
    /// let foo = String::new();
    /// ```
    #[clippy::version = "1.70.0"]
    pub REDUNDANT_TYPE_ANNOTATIONS,
    restriction,
    "warns about needless / redundant type annotations."
}
declare_lint_pass!(RedundantTypeAnnotations => [REDUNDANT_TYPE_ANNOTATIONS]);

fn is_same_type<'tcx>(cx: &LateContext<'tcx>, ty_resolved_path: hir::def::Res, func_return_type: Ty<'tcx>) -> bool {
    // type annotation is primitive
    if let hir::def::Res::PrimTy(primty) = ty_resolved_path
        && func_return_type.is_primitive()
        && let Some(func_return_type_sym) = func_return_type.primitive_symbol()
    {
        return primty.name() == func_return_type_sym;
    }

    // type annotation is any other non generic type
    if let hir::def::Res::Def(_, defid) = ty_resolved_path
        && let Some(annotation_ty) = cx.tcx.type_of(defid).no_bound_vars()
    {
        return annotation_ty == func_return_type;
    }

    false
}

fn func_hir_id_to_func_ty<'tcx>(cx: &LateContext<'tcx>, hir_id: hir::hir_id::HirId) -> Option<Ty<'tcx>> {
    if let Some((defkind, func_defid)) = cx.typeck_results().type_dependent_def(hir_id)
        && defkind == hir::def::DefKind::AssocFn
        && let Some(init_ty) = cx.tcx.type_of(func_defid).no_bound_vars()
    {
        Some(init_ty)
    } else {
        None
    }
}

fn func_ty_to_return_type<'tcx>(cx: &LateContext<'tcx>, func_ty: Ty<'tcx>) -> Option<Ty<'tcx>> {
    if func_ty.is_fn() {
        Some(func_ty.fn_sig(cx.tcx).output().skip_binder())
    } else {
        None
    }
}

/// Extracts the fn Ty, e.g. `fn() -> std::string::String {f}`
fn extract_fn_ty<'tcx>(
    cx: &LateContext<'tcx>,
    call: &hir::Expr<'tcx>,
    func_return_path: &hir::QPath<'tcx>,
) -> Option<Ty<'tcx>> {
    match func_return_path {
        // let a: String = f(); where f: fn f() -> String
        hir::QPath::Resolved(_, resolved_path) => {
            if let hir::def::Res::Def(_, defid) = resolved_path.res
                && let Some(middle_ty_init) = cx.tcx.type_of(defid).no_bound_vars()
            {
                Some(middle_ty_init)
            } else {
                None
            }
        },
        // Associated functions like
        // let a: String = String::new();
        // let a: String = String::get_string();
        hir::QPath::TypeRelative(..) => func_hir_id_to_func_ty(cx, call.hir_id),
        hir::QPath::LangItem(..) => None,
    }
}

fn is_redundant_in_func_call<'tcx>(
    cx: &LateContext<'tcx>,
    ty_resolved_path: hir::def::Res,
    call: &hir::Expr<'tcx>,
) -> bool {
    if let hir::ExprKind::Path(init_path) = &call.kind {
        let func_type = extract_fn_ty(cx, call, init_path);

        if let Some(func_type) = func_type
            && let Some(init_return_type) = func_ty_to_return_type(cx, func_type)
        {
            return is_same_type(cx, ty_resolved_path, init_return_type);
        }
    }

    false
}

fn extract_primty(ty_kind: &hir::TyKind<'_>) -> Option<hir::PrimTy> {
    if let hir::TyKind::Path(ty_path) = ty_kind
        && let hir::QPath::Resolved(_, resolved_path_ty) = ty_path
        && let hir::def::Res::PrimTy(primty) = resolved_path_ty.res
    {
        Some(primty)
    } else {
        None
    }
}

impl LateLintPass<'_> for RedundantTypeAnnotations {
    fn check_local<'tcx>(&mut self, cx: &LateContext<'tcx>, local: &'tcx rustc_hir::Local<'tcx>) {
        // type annotation part
        if !local.span.from_expansion()
            && let Some(ty) = &local.ty

            // initialization part
            && let Some(init) = local.init
        {
            match &init.kind {
                // When the initialization is a call to a function
                hir::ExprKind::Call(init_call, _) => {
                    if let hir::TyKind::Path(ty_path) = &ty.kind
                        && let hir::QPath::Resolved(_, resolved_path_ty) = ty_path

                        && is_redundant_in_func_call(cx, resolved_path_ty.res, init_call) {
                        span_lint(cx, REDUNDANT_TYPE_ANNOTATIONS, local.span, "redundant type annotation");
                    }
                },
                hir::ExprKind::MethodCall(_, _, _, _) => {
                    let mut is_ref = false;
                    let mut ty_kind = &ty.kind;

                    // If the annotation is a ref we "peel" it
                    if let hir::TyKind::Ref(_, mut_ty) = &ty.kind {
                        is_ref = true;
                        ty_kind = &mut_ty.ty.kind;
                    }

                    if let hir::TyKind::Path(ty_path) = ty_kind
                        && let hir::QPath::Resolved(_, resolved_path_ty) = ty_path
                        && let Some(func_ty) = func_hir_id_to_func_ty(cx, init.hir_id)
                        && let Some(return_type) = func_ty_to_return_type(cx, func_ty)
                        && is_same_type(cx, resolved_path_ty.res, if is_ref {
                            return_type.peel_refs()
                        } else {
                            return_type
                        })
                    {
                        span_lint(cx, REDUNDANT_TYPE_ANNOTATIONS, local.span, "redundant type annotation");
                    }
                },
                // When the initialization is a path for example u32::MAX
                hir::ExprKind::Path(init_path) => {
                    // TODO: check for non primty
                    if let Some(primty) = extract_primty(&ty.kind)

                        && let hir::QPath::TypeRelative(init_ty, _) = init_path
                        && let Some(primty_init) = extract_primty(&init_ty.kind)

                        && primty == primty_init
                    {
                        span_lint(cx, REDUNDANT_TYPE_ANNOTATIONS, local.span, "redundant type annotation");
                    }
                },
                hir::ExprKind::Lit(init_lit) => {
                    match init_lit.node {
                        // In these cases the annotation is redundant
                        LitKind::Str(..)
                        | LitKind::ByteStr(..)
                        | LitKind::Byte(..)
                        | LitKind::Char(..)
                        | LitKind::Bool(..)
                        | LitKind::CStr(..) => {
                            span_lint(cx, REDUNDANT_TYPE_ANNOTATIONS, local.span, "redundant type annotation");
                        },
                        LitKind::Int(..) | LitKind::Float(..) => {
                            // If the initialization value is a suffixed literal we lint
                            if init_lit.node.is_suffixed() {
                                span_lint(cx, REDUNDANT_TYPE_ANNOTATIONS, local.span, "redundant type annotation");
                            }
                        },
                        LitKind::Err => (),
                    }
                }
                _ => ()
            }
        };
    }
}
