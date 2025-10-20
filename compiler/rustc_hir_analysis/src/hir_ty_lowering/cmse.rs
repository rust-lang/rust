use rustc_abi::{BackendRepr, ExternAbi, Float, Integer, Primitive, Scalar};
use rustc_errors::{DiagCtxtHandle, E0781, struct_span_code_err};
use rustc_hir::{self as hir, HirId};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutError, TyAndLayout};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};
use rustc_span::Span;

use crate::errors;

/// Check conditions on inputs and outputs that the cmse ABIs impose: arguments and results MUST be
/// returned via registers (i.e. MUST NOT spill to the stack). LLVM will also validate these
/// conditions, but by checking them here rustc can emit nicer error messages.
pub(crate) fn validate_cmse_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    dcx: DiagCtxtHandle<'_>,
    hir_id: HirId,
    abi: ExternAbi,
    fn_sig: ty::PolyFnSig<'tcx>,
) {
    let fn_decl = match abi {
        ExternAbi::CmseNonSecureCall => match tcx.hir_node(hir_id) {
            hir::Node::Ty(hir::Ty { kind: hir::TyKind::FnPtr(fn_ptr_ty), .. }) => fn_ptr_ty.decl,
            _ => {
                let span = match tcx.parent_hir_node(hir_id) {
                    hir::Node::Item(hir::Item {
                        kind: hir::ItemKind::ForeignMod { .. },
                        span,
                        ..
                    }) => *span,
                    _ => tcx.hir_span(hir_id),
                };
                struct_span_code_err!(
                    dcx,
                    span,
                    E0781,
                    "the `\"cmse-nonsecure-call\"` ABI is only allowed on function pointers"
                )
                .emit();
                return;
            }
        },
        ExternAbi::CmseNonSecureEntry => {
            let Some(hir::FnSig { decl, .. }) = tcx.hir_node(hir_id).fn_sig() else {
                // might happen when this ABI is used incorrectly. That will be handled elsewhere
                return;
            };

            // An `extern "cmse-nonsecure-entry"` function cannot be c-variadic. We run
            // into https://github.com/rust-lang/rust/issues/132142 if we don't explicitly bail.
            if decl.c_variadic {
                return;
            }

            decl
        }
        _ => return,
    };

    if let Err((span, layout_err)) = is_valid_cmse_inputs(tcx, dcx, fn_sig, fn_decl, abi) {
        if should_emit_layout_error(abi, layout_err) {
            dcx.emit_err(errors::CmseGeneric { span, abi });
        }
    }

    if let Err(layout_err) = is_valid_cmse_output(tcx, dcx, fn_sig, fn_decl, abi) {
        if should_emit_layout_error(abi, layout_err) {
            dcx.emit_err(errors::CmseGeneric { span: fn_decl.output.span(), abi });
        }
    }
}

/// Returns whether the inputs will fit into the available registers
fn is_valid_cmse_inputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    dcx: DiagCtxtHandle<'_>,
    fn_sig: ty::PolyFnSig<'tcx>,
    fn_decl: &hir::FnDecl<'tcx>,
    abi: ExternAbi,
) -> Result<(), (Span, &'tcx LayoutError<'tcx>)> {
    let mut accum = 0u64;
    let mut excess_argument_spans = Vec::new();

    // this type is only used for layout computation, which does not rely on regions
    let fn_sig = tcx.instantiate_bound_regions_with_erased(fn_sig);
    let fn_sig = tcx.erase_and_anonymize_regions(fn_sig);

    for (ty, hir_ty) in fn_sig.inputs().iter().zip(fn_decl.inputs) {
        let layout = tcx
            .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(*ty))
            .map_err(|e| (hir_ty.span, e))?;

        let align = layout.layout.align().bytes();
        let size = layout.layout.size().bytes();

        accum += size;
        accum = accum.next_multiple_of(Ord::max(4, align));

        // i.e. exceeds 4 32-bit registers
        if accum > 16 {
            excess_argument_spans.push(hir_ty.span);
        }
    }

    if !excess_argument_spans.is_empty() {
        // fn f(x: u32, y: u32, z: u32, w: u16, q: u16) -> u32,
        //                                      ^^^^^^
        dcx.emit_err(errors::CmseInputsStackSpill { spans: excess_argument_spans, abi });
    }

    Ok(())
}

/// Returns whether the output will fit into the available registers
fn is_valid_cmse_output<'tcx>(
    tcx: TyCtxt<'tcx>,
    dcx: DiagCtxtHandle<'_>,
    fn_sig: ty::PolyFnSig<'tcx>,
    fn_decl: &hir::FnDecl<'tcx>,
    abi: ExternAbi,
) -> Result<(), &'tcx LayoutError<'tcx>> {
    // this type is only used for layout computation, which does not rely on regions
    let fn_sig = tcx.instantiate_bound_regions_with_erased(fn_sig);
    let fn_sig = tcx.erase_and_anonymize_regions(fn_sig);
    let return_type = fn_sig.output();

    // `impl Trait` is already disallowed with `cmse-nonsecure-call`, because that ABI is only
    // allowed on function pointers, and function pointers cannot contain `impl Trait` in their
    // signature.
    //
    // Here we explicitly disallow `impl Trait` in the `cmse-nonsecure-entry` return type too, to
    // prevent query cycles when calculating the layout. This ABI is meant to be used with
    // `#[no_mangle]` or similar, so generics in the type really don't make sense.
    //
    // see also https://github.com/rust-lang/rust/issues/147242.
    if abi == ExternAbi::CmseNonSecureEntry && return_type.has_opaque_types() {
        dcx.emit_err(errors::CmseImplTrait { span: fn_decl.output.span(), abi });
        return Ok(());
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let layout = tcx.layout_of(typing_env.as_query_input(return_type))?;

    if !is_valid_cmse_output_layout(layout) {
        dcx.emit_err(errors::CmseOutputStackSpill { span: fn_decl.output.span(), abi });
    }

    Ok(())
}

/// Returns whether the output will fit into the available registers
fn is_valid_cmse_output_layout<'tcx>(layout: TyAndLayout<'tcx>) -> bool {
    let size = layout.layout.size().bytes();

    if size <= 4 {
        return true;
    } else if size > 8 {
        return false;
    }

    // Accept scalar 64-bit types.
    let BackendRepr::Scalar(scalar) = layout.layout.backend_repr else {
        return false;
    };

    let Scalar::Initialized { value, .. } = scalar else {
        return false;
    };

    matches!(value, Primitive::Int(Integer::I64, _) | Primitive::Float(Float::F64))
}

fn should_emit_layout_error<'tcx>(abi: ExternAbi, layout_err: &'tcx LayoutError<'tcx>) -> bool {
    use LayoutError::*;

    match layout_err {
        TooGeneric(ty) => {
            match abi {
                ExternAbi::CmseNonSecureCall => {
                    // prevent double reporting of this error
                    !ty.has_opaque_types()
                }
                ExternAbi::CmseNonSecureEntry => true,
                _ => bug!("invalid ABI: {abi}"),
            }
        }
        Unknown(..)
        | SizeOverflow(..)
        | InvalidSimd { .. }
        | NormalizationFailure(..)
        | ReferencesError(..)
        | Cycle(..) => {
            false // not our job to report these
        }
    }
}
