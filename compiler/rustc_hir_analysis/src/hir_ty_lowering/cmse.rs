use rustc_errors::{DiagCtxtHandle, E0781, struct_span_code_err};
use rustc_hir::{self as hir, HirId};
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{self, ParamEnv, TyCtxt};
use rustc_span::Span;
use rustc_target::spec::abi;

use crate::errors;

/// Check conditions on inputs and outputs that the cmse ABIs impose: arguments and results MUST be
/// returned via registers (i.e. MUST NOT spill to the stack). LLVM will also validate these
/// conditions, but by checking them here rustc can emit nicer error messages.
pub(crate) fn validate_cmse_abi<'tcx>(
    tcx: TyCtxt<'tcx>,
    dcx: DiagCtxtHandle<'_>,
    hir_id: HirId,
    abi: abi::Abi,
    fn_sig: ty::PolyFnSig<'tcx>,
) {
    if let abi::Abi::CCmseNonSecureCall = abi {
        let hir_node = tcx.hir_node(hir_id);
        let hir::Node::Ty(hir::Ty {
            span: bare_fn_span,
            kind: hir::TyKind::BareFn(bare_fn_ty),
            ..
        }) = hir_node
        else {
            let span = match tcx.parent_hir_node(hir_id) {
                hir::Node::Item(hir::Item {
                    kind: hir::ItemKind::ForeignMod { .. }, span, ..
                }) => *span,
                _ => tcx.hir().span(hir_id),
            };
            struct_span_code_err!(
                tcx.dcx(),
                span,
                E0781,
                "the `\"C-cmse-nonsecure-call\"` ABI is only allowed on function pointers"
            )
            .emit();
            return;
        };

        match is_valid_cmse_inputs(tcx, fn_sig) {
            Ok(Ok(())) => {}
            Ok(Err(index)) => {
                // fn(x: u32, u32, u32, u16, y: u16) -> u32,
                //                           ^^^^^^
                let span = bare_fn_ty.param_names[index]
                    .span
                    .to(bare_fn_ty.decl.inputs[index].span)
                    .to(bare_fn_ty.decl.inputs.last().unwrap().span);
                let plural = bare_fn_ty.param_names.len() - index != 1;
                dcx.emit_err(errors::CmseCallInputsStackSpill { span, plural });
            }
            Err(layout_err) => {
                if let Some(err) = cmse_layout_err(layout_err, *bare_fn_span) {
                    dcx.emit_err(err);
                }
            }
        }

        match is_valid_cmse_output(tcx, fn_sig) {
            Ok(true) => {}
            Ok(false) => {
                let span = bare_fn_ty.decl.output.span();
                dcx.emit_err(errors::CmseCallOutputStackSpill { span });
            }
            Err(layout_err) => {
                if let Some(err) = cmse_layout_err(layout_err, *bare_fn_span) {
                    dcx.emit_err(err);
                }
            }
        };
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
        let layout = tcx.layout_of(ParamEnv::reveal_all().and(*ty))?;

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

    let mut ret_ty = fn_sig.output();
    let layout = tcx.layout_of(ParamEnv::reveal_all().and(ret_ty))?;
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
                let layout = tcx.layout_of(ParamEnv::reveal_all().and(ty))?;

                if !layout.layout.is_1zst() {
                    ret_ty = ty;
                    continue 'outer;
                }
            }
        }
    }

    Ok(ret_ty == tcx.types.i64 || ret_ty == tcx.types.u64 || ret_ty == tcx.types.f64)
}

fn cmse_layout_err<'tcx>(
    layout_err: &'tcx LayoutError<'tcx>,
    span: Span,
) -> Option<crate::errors::CmseCallGeneric> {
    use LayoutError::*;

    match layout_err {
        Unknown(ty) => {
            if ty.is_impl_trait() {
                None // prevent double reporting of this error
            } else {
                Some(errors::CmseCallGeneric { span })
            }
        }
        SizeOverflow(..) | NormalizationFailure(..) | ReferencesError(..) | Cycle(..) => {
            None // not our job to report these
        }
    }
}
