use rustc_mir::interpret::InterpErrorInfo;

use crate::*;

pub fn report_err<'tcx, 'mir>(
    ecx: &InterpCx<'mir, 'tcx, Evaluator<'tcx>>,
    mut e: InterpErrorInfo<'tcx>,
) -> Option<i64> {
    // Special treatment for some error kinds
    let msg = match e.kind {
        InterpError::MachineStop(ref info) => {
            let info = info.downcast_ref::<TerminationInfo>().expect("invalid MachineStop payload");
            match info {
                TerminationInfo::Exit(code) => return Some(*code),
                TerminationInfo::PoppedTrackedPointerTag(item) =>
                    format!("popped tracked tag for item {:?}", item),
                TerminationInfo::Abort => format!("the evaluated program aborted execution"),
            }
        }
        err_unsup!(NoMirFor(..)) => format!(
            "{}. Did you set `MIRI_SYSROOT` to a Miri-enabled sysroot? You can prepare one with `cargo miri setup`.",
            e
        ),
        InterpError::InvalidProgram(_) => bug!("This error should be impossible in Miri: {}", e),
        _ => e.to_string(),
    };
    e.print_backtrace();
    if let Some(frame) = ecx.stack().last() {
        let span = frame.current_source_info().unwrap().span;

        let msg = format!("Miri evaluation error: {}", msg);
        let mut err = ecx.tcx.sess.struct_span_err(span, msg.as_str());
        let frames = ecx.generate_stacktrace(None);
        err.span_label(span, msg);
        // We iterate with indices because we need to look at the next frame (the caller).
        for idx in 0..frames.len() {
            let frame_info = &frames[idx];
            let call_site_is_local = frames
                .get(idx + 1)
                .map_or(false, |caller_info| caller_info.instance.def_id().is_local());
            if call_site_is_local {
                err.span_note(frame_info.call_site, &frame_info.to_string());
            } else {
                err.note(&frame_info.to_string());
            }
        }
        err.emit();
    } else {
        ecx.tcx.sess.err(&msg);
    }

    for (i, frame) in ecx.stack().iter().enumerate() {
        trace!("-------------------");
        trace!("Frame {}", i);
        trace!("    return: {:?}", frame.return_place.map(|p| *p));
        for (i, local) in frame.locals.iter().enumerate() {
            trace!("    local {}: {:?}", i, local.value);
        }
    }
    // Let the reported error determine the return code.
    return None;
}

use std::cell::RefCell;
thread_local! {
    static ECX: RefCell<Vec<InterpErrorInfo<'static>>> = RefCell::new(Vec::new());
}

pub fn register_err(e: InterpErrorInfo<'static>) {
    ECX.with(|ecx| ecx.borrow_mut().push(e));
}

pub fn process_errors(mut f: impl FnMut(InterpErrorInfo<'static>)) {
    ECX.with(|ecx| {
        for e in ecx.borrow_mut().drain(..) {
            f(e);
        }
    });
}
