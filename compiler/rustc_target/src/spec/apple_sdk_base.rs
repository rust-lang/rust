use crate::spec::{cvs, LinkArgs, LinkerFlavor, LldFlavor, TargetOptions};
use std::borrow::Cow;

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
    X86_64,
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
        X86_64 | X86_64_macabi => "x86_64",
    }
}

fn target_abi(arch: Arch) -> &'static str {
    match arch {
        Armv7 | Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | X86_64 => "",
        X86_64_macabi | Arm64_macabi => "macabi",
        Arm64_sim => "sim",
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
        X86_64 => "core2",
        X86_64_macabi => "core2",
        Arm64_macabi => "apple-a12",
        Arm64_sim => "apple-a12",
    }
}

fn link_env_remove(arch: Arch) -> Cow<'static, [Cow<'static, str>]> {
    match arch {
        Armv7 | Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | X86_64 | Arm64_sim => {
            cvs!["MACOSX_DEPLOYMENT_TARGET"]
        }
        X86_64_macabi | Arm64_macabi => cvs!["IPHONEOS_DEPLOYMENT_TARGET"],
    }
}

fn pre_link_args(os: &'static str, arch: Arch) -> LinkArgs {
    let mut args = LinkArgs::new();

    let target_abi = target_abi(arch);

    let platform_name = match target_abi {
        "sim" => format!("{}-simulator", os),
        "macabi" => "mac-catalyst".to_string(),
        _ => os.to_string(),
    };

    let platform_version = match os.as_ref() {
        "ios" => super::apple_base::ios_lld_platform_version(),
        "tvos" => super::apple_base::tvos_lld_platform_version(),
        "watchos" => super::apple_base::watchos_lld_platform_version(),
        _ => unreachable!(),
    };

    let arch_str = target_arch_name(arch);

    if target_abi != "macabi" {
        args.insert(LinkerFlavor::Gcc, vec!["-arch".into(), arch_str.into()]);
    }

    args.insert(
        LinkerFlavor::Lld(LldFlavor::Ld64),
        vec![
            "-arch".into(),
            arch_str.into(),
            "-platform_version".into(),
            platform_name.into(),
            platform_version.clone().into(),
            platform_version.into(),
        ],
    );

    args
}

pub fn opts(os: &'static str, arch: Arch) -> TargetOptions {
    TargetOptions {
        abi: target_abi(arch).into(),
        cpu: target_cpu(arch).into(),
        dynamic_linking: false,
        pre_link_args: pre_link_args(os, arch),
        link_env_remove: link_env_remove(arch),
        has_thread_local: false,
        ..super::apple_base::opts(os)
    }
}
