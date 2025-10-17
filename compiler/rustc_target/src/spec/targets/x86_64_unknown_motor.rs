use crate::spec::{
    CodeModel, LinkSelfContainedDefault, LldFlavor, RelocModel, RelroLevel, Target, base,
};

pub(crate) fn target() -> Target {
    let mut base = base::motor::opts();
    base.cpu = "x86-64".into();
    base.max_atomic_width = Some(64);
    base.code_model = Some(CodeModel::Small);

    // We want fully static relocatable binaries. It was surprisingly
    // difficult to make it happen reliably, especially various
    // linker-related options below. Mostly trial and error.
    base.position_independent_executables = true;
    base.relro_level = RelroLevel::Full;
    base.static_position_independent_executables = true;
    base.relocation_model = RelocModel::Pic;
    base.lld_flavor_json = LldFlavor::Ld;
    base.link_self_contained = LinkSelfContainedDefault::True;
    base.dynamic_linking = false;
    base.crt_static_default = true;
    base.crt_static_respected = true;

    Target {
        llvm_target: "x86_64-unknown-none-elf".into(),
        metadata: crate::spec::TargetMetadata {
            description: Some("Motor OS".into()),
            tier: Some(3),
            host_tools: None,
            std: None,
        },
        pointer_width: 64,
        data_layout:
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128".into(),
        arch: "x86_64".into(),
        options: base,
    }
}
