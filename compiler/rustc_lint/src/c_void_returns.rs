use rustc_abi::ExternAbi;
use rustc_hir::def::Res;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::FnKind;
use rustc_hir::{self as hir, LangItem};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::Span;

use crate::lints::{CVoidReturn, ExternCVoidReturn};
use crate::{LateContext, LateLintPass, LintContext};

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

declare_lint_pass!(CVoidReturns => [C_VOID_RETURNS]);

impl<'tcx> LateLintPass<'tcx> for CVoidReturns {
    fn check_fn(
        &mut self,
        cx: &LateContext<'tcx>,
        fn_kind: FnKind<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
        _: &'tcx hir::Body<'tcx>,
        _: Span,
        _: LocalDefId,
    ) {
        check_decl(
            cx,
            decl,
            !matches!(fn_kind, FnKind::ItemFn(.., hir::FnHeader { abi: ExternAbi::Rust, .. })),
        );
    }

    fn check_foreign_item(&mut self, cx: &LateContext<'tcx>, item: &'tcx hir::ForeignItem<'tcx>) {
        if let hir::ForeignItemKind::Fn(sig, ..) = item.kind {
            check_decl(cx, sig.decl, true);
        }
    }

    fn check_ty(&mut self, cx: &LateContext<'tcx>, ty: &'tcx hir::Ty<'tcx, hir::AmbigArg>) {
        if let hir::TyKind::FnPtr(fn_ptr_ty) = ty.kind {
            check_decl(cx, fn_ptr_ty.decl, fn_ptr_ty.abi != ExternAbi::Rust);
        }
    }
}

fn check_decl(cx: &LateContext<'_>, decl: &hir::FnDecl<'_>, is_extern: bool) {
    if let hir::FnRetTy::Return(output_ty) = decl.output
        && let hir::TyKind::Path(qpath) = output_ty.kind
        && let Res::Def(.., def_id) = cx.qpath_res(&qpath, output_ty.hir_id)
        && cx.tcx.is_lang_item(def_id, LangItem::CVoid)
    {
        let suggestion =
            cx.sess().source_map().span_extend_to_prev_char(decl.output.span(), ')', true);

        if is_extern {
            cx.emit_span_lint(C_VOID_RETURNS, decl.output.span(), ExternCVoidReturn { suggestion });
        } else {
            cx.emit_span_lint(C_VOID_RETURNS, decl.output.span(), CVoidReturn { suggestion });
        }
    }
}
