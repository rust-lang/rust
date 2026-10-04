use crate::spec::{FramePointer, Os, TargetOptions};

pub(crate) fn opts() -> TargetOptions {
    TargetOptions {
        os: Os::SolidAsp3,
        vendor: "kmc".into(),
        executables: false,
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
