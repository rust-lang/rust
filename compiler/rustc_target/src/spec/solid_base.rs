use super::FramePointer;
use crate::spec::TargetOptions;

pub fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid_{kernel}").into(),
        vendor: "kmc".into(),
        executables: false,
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
