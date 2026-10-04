//QTEE target for aarch64

use crate::spec::{StaticCow, Target, TargetOptions, base, targets};

const LINKER_SCRIPT: &str = include_str!("./aarch64_unknown_qtee_linker_script.ld");

pub(crate) fn target() -> Target {
    Target {
        metadata: crate::spec::TargetMetadata {
            description: Some(StaticCow::Borrowed("Trusted Applications running on QTEE for ARM64")),
            tier: Some(3),
            host_tools: Some(false),
            std: None,
        },
        options: TargetOptions { link_script: Some(LINKER_SCRIPT.into()), ..base::qtee::opts() },
        ..targets::aarch64_unknown_none::target()
    }
}
