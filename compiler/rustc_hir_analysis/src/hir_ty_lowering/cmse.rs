use rustc_abi::ExternAbi;
use rustc_errors::{DiagCtxtHandle, E0781, struct_span_code_err};
use rustc_hir::{self as hir, HirId};
use rustc_middle::bug;
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{self, TyCtxt};

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
                span: bare_fn_span,
                kind: hir::TyKind::BareFn(bare_fn_ty),
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
                    let span = if let Some(ident) = bare_fn_ty.param_idents[index] {
                        ident.span.to(bare_fn_ty.decl.inputs[index].span)
                    } else {
                        bare_fn_ty.decl.inputs[index].span
                    }
                    .to(bare_fn_ty.decl.inputs.last().unwrap().span);
                    let plural = bare_fn_ty.param_idents.len() - index != 1;
                    dcx.emit_err(errors::CmseInputsStackSpill { span, plural, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseCallGeneric { span: *bare_fn_span });
                    }
                }
            }

            match is_valid_cmse_output(tcx, fn_sig) {
                Ok(true) => {}
                Ok(false) => {
                    let span = bare_fn_ty.decl.output.span();
                    dcx.emit_err(errors::CmseOutputStackSpill { span, abi });
                }
                Err(layout_err) => {
                    if should_emit_generic_error(abi, layout_err) {
                        dcx.emit_err(errors::CmseCallGeneric { span: *bare_fn_span });
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

    for (index, ty) in fn_sig.inputs().iter().enumerate() {
        let layout = tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(*ty))?;

        let align = layout.layout.align().abi.bytes();
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

    let typing_env = ty::TypingEnv::fully_monomorphized();

    let mut ret_ty = fn_sig.output();
    let layout = tcx.layout_of(typing_env.as_query_input(ret_ty))?;
    let size = layout.layout.size().bytes();

    if size <= 4 {
        return Ok(true);
    } else if size > 8 {
        return Ok(false);
    }

    // next we need to peel any repr(transparent) layers off
    'outer: loop {
        let ty::Adt(adt_def, args) = ret_ty.kind() else {
            break;
        };

        if !adt_def.repr().transparent() {
            break;
        }

        // the first field with non-trivial size and alignment must be the data
        for variant_def in adt_def.variants() {
            for field_def in variant_def.fields.iter() {
                let ty = field_def.ty(tcx, args);
                let layout = tcx.layout_of(typing_env.as_query_input(ty))?;

                if !layout.layout.is_1zst() {
                    ret_ty = ty;
                    continue 'outer;
                }
            }
        }
    }

    Ok(ret_ty == tcx.types.i64 || ret_ty == tcx.types.u64 || ret_ty == tcx.types.f64)
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
        | NormalizationFailure(..)
        | ReferencesError(..)
        | Cycle(..) => {
            false // not our job to report these
        }
    }
}
