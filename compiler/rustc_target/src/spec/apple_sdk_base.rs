use crate::spec::TargetOptions;

use Arch::*;
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum Arch {
    Armv7,
    Armv7s,
    Arm64,
    I386,
    X86_64,
    X86_64_macabi,
}

fn target_cpu(arch: Arch) -> String {
    match arch {
        Armv7 => "cortex-a8", // iOS7 is supported on iPhone 4 and higher
        Armv7s => "cortex-a9",
        Arm64 => "apple-a7",
        I386 => "yonah",
        X86_64 => "core2",
        X86_64_macabi => "core2",
    }
    .to_string()
}

fn link_env_remove(arch: Arch) -> Vec<String> {
    match arch {
        Armv7 | Armv7s | Arm64 | I386 | X86_64 => vec!["MACOSX_DEPLOYMENT_TARGET".to_string()],
        X86_64_macabi => vec!["IPHONEOS_DEPLOYMENT_TARGET".to_string()],
    }
}

pub fn opts(arch: Arch) -> TargetOptions {
    TargetOptions {
        cpu: target_cpu(arch),
        dynamic_linking: false,
        executables: true,
        link_env_remove: link_env_remove(arch),
        has_elf_tls: false,
        eliminate_frame_pointer: false,
        ..super::apple_base::opts()
    }
}
