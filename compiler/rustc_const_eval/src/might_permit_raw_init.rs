use crate::const_eval::CompileTimeInterpreter;
use crate::interpret::{InterpCx, MemoryKind, OpTy};
use rustc_middle::ty::layout::LayoutCx;
use rustc_middle::ty::{layout::TyAndLayout, ParamEnv, TyCtxt};
use rustc_session::Limit;
use rustc_target::abi::InitKind;

pub fn might_permit_raw_init<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: TyAndLayout<'tcx>,
    kind: InitKind,
) -> bool {
    let strict = tcx.sess.opts.unstable_opts.strict_init_checks;

    if strict {
        let machine = CompileTimeInterpreter::new(
            Limit::new(0),
            /*can_access_statics:*/ false,
            /*check_alignment:*/ true,
        );

        let mut cx = InterpCx::new(tcx, rustc_span::DUMMY_SP, ParamEnv::reveal_all(), machine);

        let allocated = cx
            .allocate(ty, MemoryKind::Machine(crate::const_eval::MemoryKind::Heap))
            .expect("OOM: failed to allocate for uninit check");

        if kind == InitKind::Zero {
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
