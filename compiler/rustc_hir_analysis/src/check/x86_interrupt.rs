use rustc_abi::{BackendRepr, ExternAbi, Primitive};
use rustc_errors::DiagCtxtHandle;
use rustc_hir::{self as hir};
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt};

use crate::diagnostics;

pub(crate) fn validate_x86_interrupt_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    dcx: DiagCtxtHandle<'_>,
    fn_sig: ty::FnSig<'tcx>,
    fn_decl: &hir::FnDecl<'_>,
    abi: ExternAbi,
) {
    // Rules for input validation
    // 1. 1-2 parameters (validated in AST layer)
    // 2. Neither type may be unsized -> X86InterruptUnsized
    // 3. Neither type may be zero-sized -> X86InterruptZeroSized
    // 4. First parameter (frame) must be valid for any bit pattern -> X86InterruptInvalidFrame
    // 5. First parameter (frame) must not be a pointer -> X86InterruptInvalidFrame
    // 6. Second parameter (error code) must be single word-size integer,
    //      and valid for any bit pattern -> X86InterruptInvalidErrorCode
    if abi != ExternAbi::X86Interrupt {
        return;
    }

    let get_param_layout = |(ty, _hir_ty): (Ty<'tcx>, &hir::Ty<'_>)| -> Option<TyAndLayout<'tcx>> {
        if ty.has_infer_types() {
            return None;
        }

        let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).ok()?;

        Some(layout)
    };

    let validate_frame = |frame_tys: (Ty<'tcx>, &hir::Ty<'_>)| {
        // Layout error handled elsewhere
        let Some(layout) = get_param_layout(frame_tys) else {
            return;
        };
        let (ty, hir_ty) = frame_tys;

        if layout.is_unsized() {
            dcx.emit_err(diagnostics::X86InterruptUnsized { span: hir_ty.span, ty });
            return;
        }

        if layout.is_zst() {
            dcx.emit_err(diagnostics::X86InterruptZeroSized { span: hir_ty.span, ty });
            return;
        }

        let is_pointer = match layout.backend_repr {
            BackendRepr::Scalar(s) => matches!(s.primitive(), Primitive::Pointer(_)),
            BackendRepr::ScalarPair { a: s1, b: s2, b_offset: _ } => {
                matches!(s1.primitive(), Primitive::Pointer(_))
                    || matches!(s2.primitive(), Primitive::Pointer(_))
            }
            _ => false,
        };

        if is_pointer || layout.largest_niche.is_some() {
            dcx.emit_err(diagnostics::X86InterruptInvalidFrame { span: hir_ty.span, ty });
        }
    };

    let validate_error_code = |ec_tys: (Ty<'tcx>, &hir::Ty<'_>)| {
        // Layout error handled elsewhere
        let Some(layout) = get_param_layout(ec_tys) else {
            return;
        };
        let (ty, hir_ty) = ec_tys;

        if layout.is_unsized() {
            dcx.emit_err(diagnostics::X86InterruptUnsized { span: hir_ty.span, ty });
            return;
        }

        if layout.is_zst() {
            dcx.emit_err(diagnostics::X86InterruptZeroSized { span: hir_ty.span, ty });
            return;
        }

        let ec_ok = if let BackendRepr::Scalar(scalar) = layout.backend_repr
            && let Primitive::Int(primitive, _) = scalar.primitive()
            && primitive.size() == tcx.data_layout.pointer_size()
            && scalar.is_always_valid(&tcx)
        {
            true
        } else {
            false
        };

        if !ec_ok {
            dcx.emit_err(diagnostics::X86InterruptInvalidErrorCode { span: hir_ty.span, ty });
        }
    };

    // this type is only used for layout computation, which does not rely on regions
    let fn_sig = tcx.erase_and_anonymize_regions(fn_sig);
    let param_tys: Vec<_> = fn_sig.inputs().iter().copied().zip(fn_decl.inputs).collect();

    match param_tys.as_slice() {
        [frame_tys] => {
            validate_frame(*frame_tys);
        }
        [frame_tys, ec_tys] => {
            validate_frame(*frame_tys);
            validate_error_code(*ec_tys);
        }
        _ => { /* Ignore, arity handled at AST layer. */ }
    }
}
