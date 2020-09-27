use crate::spec::{LinkArgs, LinkerFlavor, TargetOptions};
use std::env;
use std::io;
use std::path::Path;
use std::process::Command;

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

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum AppleOS {
    tvOS,
    iOS,
}

impl Arch {
    pub fn to_string(self) -> &'static str {
        match self {
            Armv7 => "armv7",
            Armv7s => "armv7s",
            Arm64 => "arm64",
            I386 => "i386",
            X86_64 => "x86_64",
            X86_64_macabi => "x86_64",
        }
    }
}

pub fn get_sdk_root(sdk_name: &str) -> Result<String, String> {
    // Following what clang does
    // (https://github.com/llvm/llvm-project/blob/
    // 296a80102a9b72c3eda80558fb78a3ed8849b341/clang/lib/Driver/ToolChains/Darwin.cpp#L1661-L1678)
    // to allow the SDK path to be set. (For clang, xcrun sets
    // SDKROOT; for rustc, the user or build system can set it, or we
    // can fall back to checking for xcrun on PATH.)
    if let Ok(sdkroot) = env::var("SDKROOT") {
        let p = Path::new(&sdkroot);
        match sdk_name {
            // Ignore `SDKROOT` if it's clearly set for the wrong platform.
            "appletvos"
                if sdkroot.contains("TVSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "appletvsimulator"
                if sdkroot.contains("TVOS.platform") || sdkroot.contains("MacOSX.platform") => {}
            "iphoneos"
                if sdkroot.contains("iPhoneSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "iphonesimulator"
                if sdkroot.contains("iPhoneOS.platform") || sdkroot.contains("MacOSX.platform") => {
            }
            "macosx10.15"
                if sdkroot.contains("iPhoneOS.platform")
                    || sdkroot.contains("iPhoneSimulator.platform") => {}
            // Ignore `SDKROOT` if it's not a valid path.
            _ if !p.is_absolute() || p == Path::new("/") || !p.exists() => {}
            _ => return Ok(sdkroot),
        }
    }
    let res =
        Command::new("xcrun").arg("--show-sdk-path").arg("-sdk").arg(sdk_name).output().and_then(
            |output| {
                if output.status.success() {
                    Ok(String::from_utf8(output.stdout).unwrap())
                } else {
                    let error = String::from_utf8(output.stderr);
                    let error = format!("process exit with error: {}", error.unwrap());
                    Err(io::Error::new(io::ErrorKind::Other, &error[..]))
                }
            },
        );

    match res {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("failed to get {} SDK path: {}", sdk_name, e)),
    }
}

fn build_pre_link_args(arch: Arch, os: AppleOS) -> Result<LinkArgs, String> {
    let sdk_name = match (arch, os) {
        (Arm64, AppleOS::tvOS) => "appletvos",
        (X86_64, AppleOS::tvOS) => "appletvsimulator",
        (Armv7, AppleOS::iOS) => "iphoneos",
        (Armv7s, AppleOS::iOS) => "iphoneos",
        (Arm64, AppleOS::iOS) => "iphoneos",
        (I386, AppleOS::iOS) => "iphonesimulator",
        (X86_64, AppleOS::iOS) => "iphonesimulator",
        (X86_64_macabi, AppleOS::iOS) => "macosx10.15",
        _ => unreachable!(),
    };

    let arch_name = arch.to_string();

    let sdk_root = get_sdk_root(sdk_name)?;

    let mut args = LinkArgs::new();
    args.insert(
        LinkerFlavor::Gcc,
        vec![
            "-arch".to_string(),
            arch_name.to_string(),
            "-isysroot".to_string(),
            sdk_root.clone(),
            "-Wl,-syslibroot".to_string(),
            sdk_root,
        ],
    );

    Ok(args)
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

pub fn opts(arch: Arch, os: AppleOS) -> Result<TargetOptions, String> {
    let pre_link_args = build_pre_link_args(arch, os)?;
    Ok(TargetOptions {
        cpu: target_cpu(arch),
        executables: true,
        pre_link_args,
        link_env_remove: link_env_remove(arch),
        has_elf_tls: false,
        eliminate_frame_pointer: false,
        ..super::apple_base::opts()
    })
}
