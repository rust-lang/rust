use crate::const_eval::CompileTimeInterpreter;
use crate::interpret::{InterpCx, MemoryKind, OpTy};
use rustc_middle::ty::layout::LayoutCx;
use rustc_middle::ty::{layout::TyAndLayout, ParamEnv, TyCtxt};
use rustc_session::Limit;
use rustc_span::Span;
use rustc_target::abi::InitKind;

pub fn might_permit_raw_init<'tcx>(
    tcx: TyCtxt<'tcx>,
    root_span: Span,
    ty: TyAndLayout<'tcx>,
    strict: bool,
    kind: InitKind,
) -> bool {
    if strict {
        let machine = CompileTimeInterpreter::new(Limit::new(0), false);

        let mut cx = InterpCx::new(tcx, root_span, ParamEnv::reveal_all(), machine);

        // We could panic here... Or we could just return "yeah it's valid whatever". Or let
        // codegen_panic_intrinsic return an error that halts compilation.
        // I'm not exactly sure *when* this can fail. OOM?
        let allocated = cx
            .allocate(ty, MemoryKind::Machine(crate::const_eval::MemoryKind::Heap))
            .expect("failed to allocate for uninit check");

        if kind == InitKind::Zero {
            // Again, unclear what to do here if it fails.
            cx.write_bytes_ptr(
                allocated.ptr,
                std::iter::repeat(0_u8).take(ty.layout.size().bytes_usize()),
            )
            .expect("failed to write bytes for zero valid check");
        }

        let ot: OpTy<'_, _> = allocated.into();

        // Assume that if it failed, it's a validation failure.
        cx.validate_operand(&ot).is_ok()
    } else {
        let layout_cx = LayoutCx { tcx, param_env: ParamEnv::reveal_all() };
        ty.might_permit_raw_init(&layout_cx, kind)
    }
}
