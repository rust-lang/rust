use rustc_middle::ty::FnSig;
use rustc_span::Span;

use crate::errors::{CmseCallInputsStackSpill, CmseCallOutputStackSpill};
use crate::traits::BuilderMethods;

/// Check conditions on inputs and outputs that the cmse ABIs impose: arguments and results MUST be
/// returned via registers (i.e. MUST NOT spill to the stack). LLVM will also validate these
/// conditions, but by checking them here rustc can emit nicer error messages.
pub fn validate_cmse_abi<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &Bx,
    fn_sig: &FnSig<'tcx>,
    call_site_span: Span,
    function_definition_span: Span,
) {
    if let rustc_target::spec::abi::Abi::CCmseNonSecureCall = fn_sig.abi {
        if !has_valid_inputs(bx, fn_sig) {
            let err = CmseCallInputsStackSpill { call_site_span, function_definition_span };
            bx.tcx().dcx().emit_err(err);
        }

        if !has_valid_output(bx, fn_sig) {
            let err = CmseCallOutputStackSpill { call_site_span, function_definition_span };
            bx.tcx().dcx().emit_err(err);
        }
    }
}

/// Returns whether the inputs will fit into the available registers
fn has_valid_inputs<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(bx: &Bx, fn_sig: &FnSig<'tcx>) -> bool {
    let mut accum = 0u64;

    for arg_def in fn_sig.inputs().iter() {
        let layout = bx.layout_of(*arg_def);

        let align = layout.layout.align().abi.bytes();
        let size = layout.layout.size().bytes();

        accum += size;
        accum = accum.next_multiple_of(Ord::max(4, align));
    }

    // the available argument space is 16 bytes (4 32-bit registers) in total
    let available_space = 16;

    accum <= available_space
}

/// Returns whether the output will fit into the available registers
fn has_valid_output<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(bx: &Bx, fn_sig: &FnSig<'tcx>) -> bool {
    let mut ret_layout = bx.layout_of(fn_sig.output());

    // unwrap any `repr(transparent)` wrappers
    loop {
        if ret_layout.is_transparent::<Bx>() {
            match ret_layout.non_1zst_field(bx) {
                None => break,
                Some((_, layout)) => ret_layout = layout,
            }
        } else {
            break;
        }
    }

    // Fundamental types of size 8 can be passed via registers according to the ABI
    let valid_2register_return_types = [bx.tcx().types.i64, bx.tcx().types.u64, bx.tcx().types.f64];

    // A Composite Type larger than 4 bytes is stored in memory at an address
    // passed as an extra argument when the function was called. That is not allowed
    // for cmse_nonsecure_entry functions.
    ret_layout.layout.size().bytes() <= 4 || valid_2register_return_types.contains(&ret_layout.ty)
}
