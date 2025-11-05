use crate::spec::{FramePointer, TargetOptions, Vendor};

pub(crate) fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid_{kernel}").into(),
        vendor: Vendor::Kmc,
        executables: false,
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
