use rustc_hir::LangItem;
use rustc_middle::mir;
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty;
use rustc_middle::ty::layout::LayoutOf;
use rustc_span::symbol::Symbol;
use rustc_type_ir::Mutability;

use crate::const_eval::{mk_eval_cx, CanAccessMutGlobal, CompileTimeEvalContext};
use crate::interpret::*;

/// Allocate a `const core::panic::Location` with the provided filename and line/column numbers.
fn alloc_caller_location<'mir, 'tcx>(
    ecx: &mut CompileTimeEvalContext<'mir, 'tcx>,
    filename: Symbol,
    line: u32,
    col: u32,
) -> MPlaceTy<'tcx> {
    let loc_details = ecx.tcx.sess.opts.unstable_opts.location_detail;
    // This can fail if rustc runs out of memory right here. Trying to emit an error would be
    // pointless, since that would require allocating more memory than these short strings.
    let file = if loc_details.file {
        ecx.allocate_str(filename.as_str(), MemoryKind::CallerLocation, Mutability::Not).unwrap()
    } else {
        // FIXME: This creates a new allocation each time. It might be preferable to
        // perform this allocation only once, and re-use the `MPlaceTy`.
        // See https://github.com/rust-lang/rust/pull/89920#discussion_r730012398
        ecx.allocate_str("<redacted>", MemoryKind::CallerLocation, Mutability::Not).unwrap()
    };
    let file = file.map_provenance(CtfeProvenance::as_immutable);
    let line = if loc_details.line { Scalar::from_u32(line) } else { Scalar::from_u32(0) };
    let col = if loc_details.column { Scalar::from_u32(col) } else { Scalar::from_u32(0) };

    // Allocate memory for `CallerLocation` struct.
    let loc_ty = ecx
        .tcx
        .type_of(ecx.tcx.require_lang_item(LangItem::PanicLocation, None))
        .instantiate(*ecx.tcx, ecx.tcx.mk_args(&[ecx.tcx.lifetimes.re_erased.into()]));
    let loc_layout = ecx.layout_of(loc_ty).unwrap();
    let location = ecx.allocate(loc_layout, MemoryKind::CallerLocation).unwrap();

    // Initialize fields.
    ecx.write_immediate(file.to_ref(ecx), &ecx.project_field(&location, 0).unwrap())
        .expect("writing to memory we just allocated cannot fail");
    ecx.write_scalar(line, &ecx.project_field(&location, 1).unwrap())
        .expect("writing to memory we just allocated cannot fail");
    ecx.write_scalar(col, &ecx.project_field(&location, 2).unwrap())
        .expect("writing to memory we just allocated cannot fail");

    location
}

pub(crate) fn const_caller_location_provider(
    tcx: TyCtxtAt<'_>,
    file: Symbol,
    line: u32,
    col: u32,
) -> mir::ConstValue<'_> {
    trace!("const_caller_location: {}:{}:{}", file, line, col);
    let mut ecx = mk_eval_cx(tcx.tcx, tcx.span, ty::ParamEnv::reveal_all(), CanAccessMutGlobal::No);

    let loc_place = alloc_caller_location(&mut ecx, file, line, col);
    if intern_const_alloc_recursive(&mut ecx, InternKind::Constant, &loc_place).is_err() {
        bug!("intern_const_alloc_recursive should not error in this case")
    }
    mir::ConstValue::Scalar(Scalar::from_maybe_pointer(loc_place.ptr(), &tcx))
}
