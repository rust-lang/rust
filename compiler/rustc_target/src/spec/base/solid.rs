use crate::spec::{FramePointer, Os, TargetOptions, Vendor};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::SolidAsp3,
        vendor: Vendor::Kmc,
        executables: false,
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
