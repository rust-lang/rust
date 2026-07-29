use rustc_abi::ExternAbi;
use rustc_hir as hir;
use rustc_hir::attrs::lang_items::LangItem;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_lint_defs::builtin::C_VOID_REFERENCES;
use rustc_lint_defs::{declare_lint, declare_lint_pass};
use rustc_middle::ty;
use rustc_span::Span;

use crate::diagnostics::{
    CVoidConst, CVoidParameter, CVoidReference, CVoidReturn, CVoidStatic, ExternCVoidReturn,
};
use crate::{LateContext, LateLintPass, LintContext};

fn is_c_void(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> bool {
    if let hir::TyKind::Path(qpath) = ty.kind
        && let Res::Def(.., def_id) = cx.qpath_res(&qpath, ty.hir_id)
    {
        // need to look through type aliases (like `std::os::raw::c_void`)
        let def_id = if DefKind::TyAlias == cx.tcx.def_kind(def_id)
            && let ty::Adt(adt_def, _) = cx.tcx.type_of(def_id).skip_binder().kind()
        {
            adt_def.did()
        } else {
            def_id
        };
        cx.tcx.is_lang_item(def_id, LangItem::CVoid)
    } else {
        false
    }
}

declare_lint! {
    /// The `c_void_parameters` lint detects the use of [`core::ffi::c_void`] as a parameter type.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::ffi::c_void;
    ///
    /// unsafe extern "C" {
    ///     fn foo(_: c_void);
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `c_void` is designed for use through a [`pointer`], equivalent to C's `void*` type. It is a
    /// mistake to use it directly as a parameter type. If you intended to use a pointer type, use
    /// `*mut c_void` or `*const c_void`.
    ///
    /// [`core::ffi::c_void`]: https://doc.rust-lang.org/core/ffi/enum.c_void.html
    /// [`pointer`]: https://doc.rust-lang.org/core/primitive.pointer.html
    /// [`()`]: https://doc.rust-lang.org/core/primitive.unit.html
    pub C_VOID_PARAMETERS,
    Warn,
    "detects use of `c_void` as a parameter type"
}

declare_lint! {
    /// The `c_void_returns` lint detects the use of [`core::ffi::c_void`] as a return type.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::ffi::c_void;
    ///
    /// unsafe extern "C" {
    ///     fn foo() -> c_void;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `c_void` is designed for use through a [`pointer`], equivalent to C's `void*` type. It is a
    /// mistake to use it directly as a return type, and calling `extern` functions declared as such
    /// may result in undefined behavior. C functions that return `void` must be declared to return
    /// [`()`] in Rust (omitting the return type implicitly returns `()`).
    ///
    /// [`core::ffi::c_void`]: https://doc.rust-lang.org/core/ffi/enum.c_void.html
    /// [`pointer`]: https://doc.rust-lang.org/core/primitive.pointer.html
    /// [`()`]: https://doc.rust-lang.org/core/primitive.unit.html
    pub C_VOID_RETURNS,
    Warn,
    "detects use of `c_void` as a return type"
}

declare_lint_pass!(CVoidParamsAndReturns => [C_VOID_PARAMETERS, C_VOID_RETURNS]);

impl<'tcx> LateLintPass<'tcx> for CVoidParamsAndReturns {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: Span,
        _: LocalDefId,
    ) {
        check_fn_decl_for_c_void(
            cx,
            decl,
            !matches!(fn_kind, FnKind::ItemFn(.., hir::FnHeader { abi: ExternAbi::Rust, .. })),
        );
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ForeignItem<'tcx>) {
        if let hir::ForeignItemKind::Fn(sig, ..) = item.kind {
            check_fn_decl_for_c_void(cx, sig.decl, true);
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Fn(sig, ..) = item.kind {
            check_fn_decl_for_c_void(cx, sig.decl, false);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) {
        if let hir::TyKind::FnPtr(fn_ptr_ty) = ty.kind {
            check_fn_decl_for_c_void(cx, fn_ptr_ty.decl, fn_ptr_ty.abi != ExternAbi::Rust);
        }
    }
}

fn check_fn_decl_for_c_void(cx: &LateContext<'_>, decl: &hir::FnDecl<'_>, is_extern: bool) {
    if let hir::FnRetTy::Return(output_ty) = decl.output
        && is_c_void(cx, output_ty)
    {
        let suggestion =
            cx.sess().source_map().span_extend_to_prev_char(decl.output.span(), ')', true);

        if is_extern {
            cx.emit_span_lint(C_VOID_RETURNS, decl.output.span(), ExternCVoidReturn { suggestion });
        } else {
            cx.emit_span_lint(C_VOID_RETURNS, decl.output.span(), CVoidReturn { suggestion });
        }
    }

    for param_ty in decl.inputs {
        if is_c_void(cx, param_ty) {
            cx.emit_span_lint(C_VOID_PARAMETERS, param_ty.span, CVoidParameter);
        }
    }
}

declare_lint_pass!(CVoidReferences => [C_VOID_REFERENCES]);

impl<'tcx> LateLintPass<'tcx> for CVoidReferences {
    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) {
        if let hir::TyKind::Ref(_, hir::MutTy { ty: pointee_ty, .. }) = ty.kind
            && is_c_void(cx, pointee_ty)
        {
            cx.emit_span_lint(C_VOID_REFERENCES, ty.span, CVoidReference);
        }
    }
}

declare_lint! {
    /// The `c_void_statics` lint detects the use of [`core::ffi::c_void`] as the type of a `static` or `const` item.
    ///
    /// ### Example
    ///
    /// ```rust
    /// use std::ffi::c_void;
    ///
    /// unsafe extern "C" {
    ///     static FOO: c_void;
    /// }
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// `c_void` is designed for use through a [`raw pointer`], equivalent to C's `void*` type.
    /// For historical reasons, Rust considers it to have size 1. For a `static` item used
    /// only for its address, the correct type is [`()`].
    ///
    /// [`core::ffi::c_void`]: https://doc.rust-lang.org/core/ffi/enum.c_void.html
    /// [`raw pointer`]: https://doc.rust-lang.org/core/primitive.pointer.html
    /// [`()`]: https://doc.rust-lang.org/core/primitive.unit.html
    pub C_VOID_STATICS,
    Warn,
    "detects use of `c_void` as the type of a `static` or `const` item"
}

declare_lint_pass!(CVoidStatics => [C_VOID_STATICS]);

impl<'tcx> LateLintPass<'tcx> for CVoidStatics {
    fn check_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx rustc_hir::Item<'tcx>) {
        match item.kind {
            hir::ItemKind::Static(_mut, _name, ty, _) if is_c_void(cx, ty) => {
                cx.emit_span_lint(C_VOID_STATICS, item.span, CVoidStatic { suggestion: ty.span })
            }
            hir::ItemKind::Const(_name, _generics, ty, _) if is_c_void(cx, ty) => {
                cx.emit_span_lint(C_VOID_STATICS, item.span, CVoidConst)
            }
            _ => (),
        }
    }

    fn check_foreign_item(
        &mut self,
        cx: &LateContext<'tcx>,
        item: &'tcx rustc_hir::ForeignItem<'tcx>,
    ) {
        if let hir::ForeignItemKind::Static(ty, ..) = item.kind
            && is_c_void(cx, ty)
        {
            cx.emit_span_lint(C_VOID_STATICS, item.span, CVoidStatic { suggestion: ty.span })
        }
    }

    fn check_trait_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx rustc_hir::TraitItem<'tcx>) {
        if let hir::TraitItemKind::Const(ty, ..) = item.kind
            && is_c_void(cx, ty)
        {
            cx.emit_span_lint(C_VOID_STATICS, item.span, CVoidConst)
        }
    }
}
