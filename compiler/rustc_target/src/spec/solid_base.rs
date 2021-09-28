use super::FramePointer;
use crate::spec::TargetOptions;

pub fn opts(kernel: &str) -> TargetOptions {
    TargetOptions {
        os: format!("solid_{}", kernel),
        vendor: "kmc".to_string(),
        frame_pointer: FramePointer::NonLeaf,
        has_elf_tls: true,
        ..Default::default()
    }
}
