use super::FramePointer;
use crate::spec::TargetOptions;

pub fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid_{}", kernel),
        vendor: "kmc".to_string(),
        frame_pointer: FramePointer::NonLeaf,
        has_thread_local: true,
        ..Default::default()
    }
}
