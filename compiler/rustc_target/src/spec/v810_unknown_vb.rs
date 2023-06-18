use crate::spec::{Target, TargetOptions};

pub fn target() -> Target {
    let base = opts();
    Target {
        llvm_target: "v810-unknown-vb".into(),
        pointer_width: 32,
        data_layout: "e-p:32:16-i32:32".into(),
        arch: "v810".into(),
        options: base,
    }
}

fn opts() -> TargetOptions {
    let mut options: TargetOptions = Default::default();
    options.cpu = "vb".into();
    options
}