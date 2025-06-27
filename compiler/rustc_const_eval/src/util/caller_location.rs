use rustc_abi::FieldIdx;
use rustc_hir::LangItem;
use rustc_middle::ty::{self, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_span::Symbol;
use tracing::trace;

use crate::const_eval::{CanAccessMutGlobal, CompileTimeInterpCx, mk_eval_cx_to_read_const_val};
use crate::interpret::*;

/// Allocate a `const core::panic::Location` with the provided filename and line/column numbers.
fn alloc_caller_location<'tcx>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    filename: Symbol,
    line: u32,
    col: u32,
) -> MPlaceTy<'tcx> {
    // Ensure that the filename itself does not contain nul bytes.
    // This isn't possible via POSIX or Windows, but we should ensure no one
    // ever does such a thing.
    assert!(!filename.as_str().as_bytes().contains(&0));

    let loc_details = ecx.tcx.sess.opts.unstable_opts.location_detail;
    let file_wide_ptr = {
        let filename = if loc_details.file { filename.as_str() } else { "<redacted>" };
        let filename_with_nul = filename.to_owned() + "\0";
        // This can fail if rustc runs out of memory right here. Trying to emit an error would be
        // pointless, since that would require allocating more memory than these short strings.
        let file_ptr = ecx.allocate_bytes_dedup(filename_with_nul.as_bytes()).unwrap();
        Immediate::new_slice(file_ptr.into(), filename_with_nul.len().try_into().unwrap(), ecx)
    };
    let line = if loc_details.line { Scalar::from_u32(line) } else { Scalar::from_u32(0) };
    let col = if loc_details.column { Scalar::from_u32(col) } else { Scalar::from_u32(0) };

    // Allocate memory for `CallerLocation` struct.
    let loc_ty = ecx
        .tcx
        .type_of(ecx.tcx.require_lang_item(LangItem::PanicLocation, ecx.tcx.span))
        .instantiate(*ecx.tcx, ecx.tcx.mk_args(&[ecx.tcx.lifetimes.re_erased.into()]));
    let loc_layout = ecx.layout_of(loc_ty).unwrap();
    let location = ecx.allocate(loc_layout, MemoryKind::CallerLocation).unwrap();

    // Initialize fields.
    ecx.write_immediate(
        file_wide_ptr,
        &ecx.project_field(&location, FieldIdx::from_u32(0)).unwrap(),
    )
    .expect("writing to memory we just allocated cannot fail");
    ecx.write_scalar(line, &ecx.project_field(&location, FieldIdx::from_u32(1)).unwrap())
        .expect("writing to memory we just allocated cannot fail");
    ecx.write_scalar(col, &ecx.project_field(&location, FieldIdx::from_u32(2)).unwrap())
        .expect("writing to memory we just allocated cannot fail");

    location
}

pub(crate) fn const_caller_location_provider(
    tcx: TyCtxt<'_>,
    file: Symbol,
    line: u32,
    col: u32,
) -> mir::ConstValue<'_> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx_to_read_const_val(
        tcx,
        rustc_span::DUMMY_SP, // FIXME: use a proper span here?
        ty::TypingEnv::fully_monomorphized(),
        CanAccessMutGlobal::No,
    );

    let loc_place = alloc_caller_location(&mut ecx, file, line, col);
    if intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &loc_place).is_err() {
        bug!("intern_const_alloc_recursive should not error in this case")
    }
    mir::ConstValue::Scalar(Scalar::from_maybe_pointer(loc_place.ptr(), &tcx))
}
