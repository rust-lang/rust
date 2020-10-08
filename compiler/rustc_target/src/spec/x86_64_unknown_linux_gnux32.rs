use crate::spec::{LinkerFlavor, Target};

pub fn target() -> Target {
    let mut base = super::linux_base::opts();
    base.cpu = "x86-64".to_string();
    base.max_atomic_width = Some(64);
    base.pre_link_args.get_mut(&LinkerFlavor::Gcc).unwrap().push("-mx32".to_string());
    base.stack_probes = true;
    base.has_elf_tls = false;
    // BUG(GabrielMajeri): disabling the PLT on x86_64 Linux with x32 ABI
    // breaks code gen. See LLVM bug 36743
    base.needs_plt = true;

    Target {
        llvm_target: "x86_64-unknown-linux-gnux32".to_string(),
        pointer_width: 32,
        data_layout: "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-\
            i64:64-f80:128-n8:16:32:64-S128"
            .to_string(),
        arch: "x86_64".to_string(),
        options: base,
    }
}
