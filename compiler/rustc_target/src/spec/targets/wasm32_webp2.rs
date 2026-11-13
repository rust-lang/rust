use crate::spec::{Env, LinkSelfContainedDefault, Os, RelocModel, Target};

pub(crate) fn target() -> Target {
    // This target is derived structurally from the wasip2 target as it's pretty
    // similar toolchain-wise, but some parts are configured to be different.
    let mut target = super::wasm32_wasip2::target();
    target.llvm_target = "wasm32-webp2".into();
    target.metadata = crate::spec::TargetMetadata {
        description: Some("WebAssembly".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };
    target.options.os = Os::Web;
    target.options.env = Env::P2;

    // As a tier 3 target there's nothing self-contained at this time unlike the
    // wasip2 target. This might change in the future should webp2 become a tier
    // 2 target.
    target.options.link_self_contained = LinkSelfContainedDefault::False;

    // This target isn't expected to be used with dynamic libraries to the same
    // degree wasip2 is, and binary size is more of a concern on the web, so go
    // back to the base wasm target's default of static reloc model.
    target.options.relocation_model = RelocModel::Static;

    target
}
