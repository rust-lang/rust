use rustc_abi::{BackendRepr, ExternAbi, Float, Integer, Primitive, Scalar};
use rustc_errors::{DiagCtxtHandle, E0781, struct_span_code_err};
use rustc_hir::{self as hir, HirId};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutError, TyAndLayout};
use rustc_middle::ty::{self, TyCtxt, TypeVisitableExt};

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
    match abi {
        ExternAbi::CmseNonSecureCall => {
            let hir_node = tcx.hir_node(hir_id);
            let hir::Node::Ty(hir::Ty {
                span: fn_ptr_span,
                kind: hir::TyKind::FnPtr(fn_ptr_ty),
                ..
            }) = hir_node
            else {
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
            };

            match is_valid_cmse_inputs(tcx, fn_sig) {
                Ok(Ok(())) => {}
                Ok(Err(index)) => {
                    // fn(x: u32, u32, u32, u16, y: u16) -> u32,
                    //                           ^^^^^^
                    let span = if let Some(ident) = fn_ptr_ty.param_idents[index] {
                        ident.span.to(fn_ptr_ty.decl.inputs[index].span)
                    } else {
                        fn_ptr_ty.decl.inputs[index].span
                    }
                    .to(fn_ptr_ty.decl.inputs.last().unwrap().span);
                    let plural = fn_ptr_ty.param_idents.len() - index != 1;
                    dcx.emit_err(errors::CmseInputsStackSpill { span, plural, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseCallGeneric { span: *fn_ptr_span });
                    }
                }
            }

            match is_valid_cmse_output(tcx, fn_sig) {
                Ok(true) => {}
                Ok(false) => {
                    let span = fn_ptr_ty.decl.output.span();
                    dcx.emit_err(errors::CmseOutputStackSpill { span, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseCallGeneric { span: *fn_ptr_span });
                    }
                }
            };
        }
        ExternAbi::CmseNonSecureEntry => {
            let hir_node = tcx.hir_node(hir_id);
            let Some(hir::FnSig { decl, span: fn_sig_span, .. }) = hir_node.fn_sig() else {
                // might happen when this ABI is used incorrectly. That will be handled elsewhere
                return;
            };

            // An `extern "cmse-nonsecure-entry"` function cannot be c-variadic. We run
            // into https://github.com/rust-lang/rust/issues/132142 if we don't explicitly bail.
            if decl.c_variadic {
                return;
            }

            match is_valid_cmse_inputs(tcx, fn_sig) {
                Ok(Ok(())) => {}
                Ok(Err(index)) => {
                    // fn f(x: u32, y: u32, z: u32, w: u16, q: u16) -> u32,
                    //                                      ^^^^^^
                    let span = decl.inputs[index].span.to(decl.inputs.last().unwrap().span);
                    let plural = decl.inputs.len() - index != 1;
                    dcx.emit_err(errors::CmseInputsStackSpill { span, plural, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseEntryGeneric { span: *fn_sig_span });
                    }
                }
            }

            match is_valid_cmse_output(tcx, fn_sig) {
                Ok(true) => {}
                Ok(false) => {
                    let span = decl.output.span();
                    dcx.emit_err(errors::CmseOutputStackSpill { span, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseEntryGeneric { span: *fn_sig_span });
                    }
                }
            };
        }
        _ => (),
    }
}

/// Returns whether the inputs will fit into the available registers
fn is_valid_cmse_inputs<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
) -> Result<Result<(), usize>, &'tcx LayoutError<'tcx>> {
    let mut span = None;
    let mut accum = 0u64;

    // this type is only used for layout computation, which does not rely on regions
    let fn_sig = tcx.instantiate_bound_regions_with_erased(fn_sig);
    let fn_sig = tcx.erase_and_anonymize_regions(fn_sig);

    for (index, ty) in fn_sig.inputs().iter().enumerate() {
        let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(*ty))?;

        let align = layout.layout.align().bytes();
        let size = layout.layout.size().bytes();

        accum += size;
        accum = accum.next_multiple_of(Ord::max(4, align));

        // i.e. exceeds 4 32-bit registers
        if accum > 16 {
            span = span.or(Some(index));
        }
    }

    match span {
        None => Ok(Ok(())),
        Some(span) => Ok(Err(span)),
    }
}

/// Returns whether the output will fit into the available registers
fn is_valid_cmse_output<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_sig: ty::PolyFnSig<'tcx>,
) -> Result<bool, &'tcx LayoutError<'tcx>> {
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
    if return_type.has_opaque_types() {
        return Err(tcx.arena.alloc(LayoutError::TooGeneric(return_type)));
    }

    let typing_env = ty::TypingEnv::fully_monomorphized();
    let layout = tcx.layout_of(typing_env.as_query_input(return_type))?;

    Ok(is_valid_cmse_output_layout(layout))
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

fn should_emit_generic_error<'tcx>(abi: ExternAbi, layout_err: &'tcx LayoutError<'tcx>) -> bool {
    use LayoutError::*;

    match layout_err {
        TooGeneric(ty) => {
            match abi {
                ExternAbi::CmseNonSecureCall => {
                    // prevent double reporting of this error
                    !ty.is_impl_trait()
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
