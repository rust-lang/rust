use crate::spec::{cvs, TargetOptions};
use std::borrow::Cow;

#[cfg(test)]
#[path = "apple/tests.rs"]
mod tests;

use Arch::*;
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum Arch {
    Armv7,
    Armv7k,
    Armv7s,
    Arm64,
    Arm64_32,
    I386,
    #[allow(dead_code)] // Some targets don't use this enum...
    X86_64,
    X86_64_sim,
    X86_64_macabi,
    Arm64_macabi,
    Arm64_sim,
}

fn target_arch_name(arch: Arch) -> &'static str {
    match arch {
        Armv7 => "armv7",
        Armv7k => "armv7k",
        Armv7s => "armv7s",
        Arm64 | Arm64_macabi | Arm64_sim => "arm64",
        Arm64_32 => "arm64_32",
        I386 => "i386",
        X86_64 | X86_64_sim | X86_64_macabi => "x86_64",
    }
}

fn target_abi(arch: Arch) -> &'static str {
    match arch {
        Armv7 | Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | X86_64 => "",
        X86_64_macabi | Arm64_macabi => "macabi",
        // x86_64-apple-ios is a simulator target, even though it isn't
        // declared that way in the target like the other ones...
        Arm64_sim | X86_64_sim => "sim",
    }
}

fn target_cpu(arch: Arch) -> &'static str {
    match arch {
        Armv7 => "cortex-a8", // iOS7 is supported on iPhone 4 and higher
        Armv7k => "cortex-a8",
        Armv7s => "cortex-a9",
        Arm64 => "apple-a7",
        Arm64_32 => "apple-s4",
        I386 => "yonah",
        X86_64 | X86_64_sim => "core2",
        X86_64_macabi => "core2",
        Arm64_macabi => "apple-a12",
        Arm64_sim => "apple-a12",
    }
}

fn link_env_remove(arch: Arch) -> Cow<'static, [Cow<'static, str>]> {
    match arch {
        Armv7 | Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | X86_64 | X86_64_sim | Arm64_sim => {
            cvs!["MACOSX_DEPLOYMENT_TARGET"]
        }
        X86_64_macabi | Arm64_macabi => cvs!["IPHONEOS_DEPLOYMENT_TARGET"],
    }
}

pub fn opts(os: &'static str, arch: Arch) -> TargetOptions {
    let abi = target_abi(arch);
    TargetOptions {
        abi: abi.into(),
        cpu: target_cpu(arch).into(),
        link_env_remove: link_env_remove(arch),
        has_thread_local: false,
        ..super::apple_base::opts(os, target_arch_name(arch), abi)
    }
}
