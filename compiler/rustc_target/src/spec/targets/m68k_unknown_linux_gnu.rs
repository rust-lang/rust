use rustc_abi::Endian;

use crate::spec::{LinkSelfContainedDefault, Target, TargetMetadata, TargetOptions, base};

pub(crate) fn target() -> Target {
    let mut base = base::linux_gnu::opts();
    base.cpu = "M68020".into();
    base.max_atomic_width = Some(32);

    Target {
        llvm_target: "m68k-unknown-linux-gnu".into(),
        metadata: TargetMetadata {
            description: Some("Motorola 680x0 Linux".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(true),
        },
        pointer_width: 32,
        data_layout: "E-m:e-p:32:16:32-i8:8:8-i16:16:16-i32:16:32-n8:16:32-a:0:16-S16".into(),
        arch: "m68k".into(),
        options: TargetOptions {
            endian: Endian::Big,
            mcount: "_mcount".into(),

            // LLD currently does not have support for M68k
            link_self_contained: LinkSelfContainedDefault::False,
            ..base
        },
    }
}
