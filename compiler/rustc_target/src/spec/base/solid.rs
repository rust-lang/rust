use crate::spec::{FramePointer, TargetOptions};

pub(crate) fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid_{kernel}").into(),
        vendor: "kmc".into(),
        executables: false,
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
