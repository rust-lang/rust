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
    Arm64_macabi,
    Arm64_sim,
}

fn target_abi(arch: Arch) -> String {
    match arch {
        Armv7 | Armv7s | Arm64 | I386 | X86_64 => "",
        X86_64_macabi | Arm64_macabi => "macabi",
        Arm64_sim => "sim",
    }
    .to_string()
}

fn target_cpu(arch: Arch) -> String {
    match arch {
        Armv7 => "cortex-a8", // iOS7 is supported on iPhone 4 and higher
        Armv7s => "cortex-a9",
        Arm64 => "apple-a7",
        I386 => "yonah",
        X86_64 => "core2",
        X86_64_macabi => "core2",
        Arm64_macabi => "apple-a12",
        Arm64_sim => "apple-a12",
    }
    .to_string()
}

fn link_env_remove(arch: Arch) -> Vec<String> {
    match arch {
        Armv7 | Armv7s | Arm64 | I386 | X86_64 | Arm64_sim => {
            vec!["MACOSX_DEPLOYMENT_TARGET".to_string()]
        }
        X86_64_macabi | Arm64_macabi => vec!["IPHONEOS_DEPLOYMENT_TARGET".to_string()],
    }
}

pub fn opts(os: &str, arch: Arch) -> TargetOptions {
    TargetOptions {
        abi: target_abi(arch),
        cpu: target_cpu(arch),
        dynamic_linking: false,
        executables: true,
        link_env_remove: link_env_remove(arch),
        has_thread_local: false,
        ..super::apple_base::opts(os)
    }
}
