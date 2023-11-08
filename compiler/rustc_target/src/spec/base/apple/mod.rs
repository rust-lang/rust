use std::{borrow::Cow, env};

use crate::spec::{add_link_args, add_link_args_iter};
use crate::spec::{cvs, Cc, DebuginfoKind, FramePointer, LinkArgs};
use crate::spec::{LinkerFlavor, Lld, SplitDebuginfo, StaticCow, Target, TargetOptions};

#[cfg(test)]
mod tests;

use Arch::*;
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub enum Arch {
    Armv7k,
    Armv7s,
    Arm64,
    Arm64_32,
    I386,
    I686,
    X86_64,
    X86_64h,
    X86_64_sim,
    X86_64_macabi,
    Arm64_macabi,
    Arm64_sim,
}

impl Arch {
    pub fn target_name(self) -> &'static str {
        match self {
            Armv7k => "armv7k",
            Armv7s => "armv7s",
            Arm64 | Arm64_macabi | Arm64_sim => "arm64",
            Arm64_32 => "arm64_32",
            I386 => "i386",
            I686 => "i686",
            X86_64 | X86_64_sim | X86_64_macabi => "x86_64",
            X86_64h => "x86_64h",
        }
    }

    pub fn target_arch(self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            Armv7k | Armv7s => "arm",
            Arm64 | Arm64_32 | Arm64_macabi | Arm64_sim => "aarch64",
            I386 | I686 => "x86",
            X86_64 | X86_64_sim | X86_64_macabi | X86_64h => "x86_64",
        })
    }

    fn target_abi(self) -> &'static str {
        match self {
            Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | I686 | X86_64 | X86_64h => "",
            X86_64_macabi | Arm64_macabi => "macabi",
            // x86_64-apple-ios is a simulator target, even though it isn't
            // declared that way in the target like the other ones...
            Arm64_sim | X86_64_sim => "sim",
        }
    }

    fn target_cpu(self) -> &'static str {
        match self {
            Armv7k => "cortex-a8",
            Armv7s => "swift", // iOS 10 is only supported on iPhone 5 or higher.
            Arm64 => "apple-a7",
            Arm64_32 => "apple-s4",
            // Only macOS 10.12+ is supported, which means
            // all x86_64/x86 CPUs must be running at least penryn
            // https://github.com/llvm/llvm-project/blob/01f924d0e37a5deae51df0d77e10a15b63aa0c0f/clang/lib/Driver/ToolChains/Arch/X86.cpp#L79-L82
            I386 | I686 => "penryn",
            X86_64 | X86_64_sim => "penryn",
            X86_64_macabi => "penryn",
            // Note: `core-avx2` is slightly more advanced than `x86_64h`, see
            // comments (and disabled features) in `x86_64h_apple_darwin` for
            // details. It is a higher baseline then `penryn` however.
            X86_64h => "core-avx2",
            Arm64_macabi => "apple-a12",
            Arm64_sim => "apple-a12",
        }
    }
}

fn pre_link_args(os: &'static str, arch: Arch, abi: &'static str) -> LinkArgs {
    let platform_name: StaticCow<str> = match abi {
        "sim" => format!("{os}-simulator").into(),
        "macabi" => "mac-catalyst".into(),
        _ => os.into(),
    };

    let platform_version: StaticCow<str> = match os {
        "ios" => ios_lld_platform_version(),
        "tvos" => tvos_lld_platform_version(),
        "watchos" => watchos_lld_platform_version(),
        "macos" => macos_lld_platform_version(arch),
        _ => unreachable!(),
    }
    .into();

    let arch = arch.target_name();

    let mut args = TargetOptions::link_args(
        LinkerFlavor::Darwin(Cc::No, Lld::No),
        &["-arch", arch, "-platform_version"],
    );
    add_link_args_iter(
        &mut args,
        LinkerFlavor::Darwin(Cc::No, Lld::No),
        [platform_name, platform_version.clone(), platform_version].into_iter(),
    );
    if abi != "macabi" {
        add_link_args(&mut args, LinkerFlavor::Darwin(Cc::Yes, Lld::No), &["-arch", arch]);
    }

    args
}

pub fn opts(os: &'static str, arch: Arch) -> TargetOptions {
    let abi = arch.target_abi();

    TargetOptions {
        abi: abi.into(),
        os: os.into(),
        cpu: arch.target_cpu().into(),
        link_env_remove: link_env_remove(arch, os),
        vendor: "apple".into(),
        linker_flavor: LinkerFlavor::Darwin(Cc::Yes, Lld::No),
        // macOS has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        pre_link_args: pre_link_args(os, arch, abi),
        families: cvs!["unix"],
        is_like_osx: true,
        // LLVM notes that macOS 10.11+ and iOS 9+ default
        // to v4, so we do the same.
        // https://github.com/llvm/llvm-project/blob/378778a0d10c2f8d5df8ceff81f95b6002984a4b/clang/lib/Driver/ToolChains/Darwin.cpp#L1203
        default_dwarf_version: 4,
        frame_pointer: FramePointer::Always,
        has_rpath: true,
        dll_suffix: ".dylib".into(),
        archive_format: "darwin".into(),
        // Thread locals became available with iOS 8 and macOS 10.7,
        // and both are far below our minimum.
        has_thread_local: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        eh_frame_header: false,

        debuginfo_kind: DebuginfoKind::DwarfDsym,
        // The historical default for macOS targets is to run `dsymutil` which
        // generates a packed version of debuginfo split from the main file.
        split_debuginfo: SplitDebuginfo::Packed,
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebuginfo::Packed,
            SplitDebuginfo::Unpacked,
            SplitDebuginfo::Off,
        ]),

        // This environment variable is pretty magical but is intended for
        // producing deterministic builds. This was first discovered to be used
        // by the `ar` tool as a way to control whether or not mtime entries in
        // the archive headers were set to zero or not. It appears that
        // eventually the linker got updated to do the same thing and now reads
        // this environment variable too in recent versions.
        //
        // For some more info see the commentary on #47086
        link_env: Cow::Borrowed(&[(Cow::Borrowed("ZERO_AR_DATE"), Cow::Borrowed("1"))]),

        ..Default::default()
    }
}

pub fn sdk_version(platform: u32) -> Option<(u32, u32)> {
    // NOTE: These values are from an arbitrary point in time but shouldn't make it into the final
    // binary since the final link command will have the current SDK version passed to it.
    match platform {
        object::macho::PLATFORM_MACOS => Some((13, 1)),
        object::macho::PLATFORM_IOS
        | object::macho::PLATFORM_IOSSIMULATOR
        | object::macho::PLATFORM_TVOS
        | object::macho::PLATFORM_TVOSSIMULATOR
        | object::macho::PLATFORM_MACCATALYST => Some((16, 2)),
        object::macho::PLATFORM_WATCHOS | object::macho::PLATFORM_WATCHOSSIMULATOR => Some((9, 1)),
        _ => None,
    }
}

pub fn platform(target: &Target) -> Option<u32> {
    Some(match (&*target.os, &*target.abi) {
        ("macos", _) => object::macho::PLATFORM_MACOS,
        ("ios", "macabi") => object::macho::PLATFORM_MACCATALYST,
        ("ios", "sim") => object::macho::PLATFORM_IOSSIMULATOR,
        ("ios", _) => object::macho::PLATFORM_IOS,
        ("watchos", "sim") => object::macho::PLATFORM_WATCHOSSIMULATOR,
        ("watchos", _) => object::macho::PLATFORM_WATCHOS,
        ("tvos", "sim") => object::macho::PLATFORM_TVOSSIMULATOR,
        ("tvos", _) => object::macho::PLATFORM_TVOS,
        _ => return None,
    })
}

pub fn deployment_target(target: &Target) -> Option<(u32, u32)> {
    let (major, minor) = match &*target.os {
        "macos" => {
            // This does not need to be specific. It just needs to handle x86 vs M1.
            let arch = if target.arch == "x86" || target.arch == "x86_64" { X86_64 } else { Arm64 };
            macos_deployment_target(arch)
        }
        "ios" => match &*target.abi {
            "macabi" => mac_catalyst_deployment_target(),
            _ => ios_deployment_target(),
        },
        "watchos" => watchos_deployment_target(),
        "tvos" => tvos_deployment_target(),
        _ => return None,
    };

    Some((major, minor))
}

fn from_set_deployment_target(var_name: &str) -> Option<(u32, u32)> {
    let deployment_target = env::var(var_name).ok()?;
    let (unparsed_major, unparsed_minor) = deployment_target.split_once('.')?;
    let (major, minor) = (unparsed_major.parse().ok()?, unparsed_minor.parse().ok()?);

    Some((major, minor))
}

fn macos_default_deployment_target(arch: Arch) -> (u32, u32) {
    match arch {
        // Note: Arm64_sim is not included since macOS has no simulator.
        Arm64 | Arm64_macabi => (11, 0),
        _ => (10, 12),
    }
}

fn macos_deployment_target(arch: Arch) -> (u32, u32) {
    // If you are looking for the default deployment target, prefer `rustc --print deployment-target`.
    from_set_deployment_target("MACOSX_DEPLOYMENT_TARGET")
        .unwrap_or_else(|| macos_default_deployment_target(arch))
}

fn macos_lld_platform_version(arch: Arch) -> String {
    let (major, minor) = macos_deployment_target(arch);
    format!("{major}.{minor}")
}

pub fn macos_llvm_target(arch: Arch) -> String {
    let (major, minor) = macos_deployment_target(arch);
    format!("{}-apple-macosx{}.{}.0", arch.target_name(), major, minor)
}

fn link_env_remove(arch: Arch, os: &'static str) -> StaticCow<[StaticCow<str>]> {
    // Apple platforms only officially support macOS as a host for any compilation.
    //
    // If building for macOS, we go ahead and remove any erroneous environment state
    // that's only applicable to cross-OS compilation. Always leave anything for the
    // host OS alone though.
    if os == "macos" {
        let mut env_remove = Vec::with_capacity(2);
        // Remove the `SDKROOT` environment variable if it's clearly set for the wrong platform, which
        // may occur when we're linking a custom build script while targeting iOS for example.
        if let Ok(sdkroot) = env::var("SDKROOT") {
            if sdkroot.contains("iPhoneOS.platform")
                || sdkroot.contains("iPhoneSimulator.platform")
                || sdkroot.contains("AppleTVOS.platform")
                || sdkroot.contains("AppleTVSimulator.platform")
                || sdkroot.contains("WatchOS.platform")
                || sdkroot.contains("WatchSimulator.platform")
            {
                env_remove.push("SDKROOT".into())
            }
        }
        // Additionally, `IPHONEOS_DEPLOYMENT_TARGET` must not be set when using the Xcode linker at
        // "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld",
        // although this is apparently ignored when using the linker at "/usr/bin/ld".
        env_remove.push("IPHONEOS_DEPLOYMENT_TARGET".into());
        env_remove.push("TVOS_DEPLOYMENT_TARGET".into());
        env_remove.into()
    } else {
        // Otherwise if cross-compiling for a different OS/SDK, remove any part
        // of the linking environment that's wrong and reversed.
        match arch {
            Armv7k | Armv7s | Arm64 | Arm64_32 | I386 | I686 | X86_64 | X86_64_sim | X86_64h
            | Arm64_sim => {
                cvs!["MACOSX_DEPLOYMENT_TARGET"]
            }
            X86_64_macabi | Arm64_macabi => cvs!["IPHONEOS_DEPLOYMENT_TARGET"],
        }
    }
}

fn ios_deployment_target() -> (u32, u32) {
    // If you are looking for the default deployment target, prefer `rustc --print deployment-target`.
    from_set_deployment_target("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or((10, 0))
}

fn mac_catalyst_deployment_target() -> (u32, u32) {
    // If you are looking for the default deployment target, prefer `rustc --print deployment-target`.
    from_set_deployment_target("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or((14, 0))
}

pub fn ios_llvm_target(arch: Arch) -> String {
    // Modern iOS tooling extracts information about deployment target
    // from LC_BUILD_VERSION. This load command will only be emitted when
    // we build with a version specific `llvm_target`, with the version
    // set high enough. Luckily one LC_BUILD_VERSION is enough, for Xcode
    // to pick it up (since std and core are still built with the fallback
    // of version 7.0 and hence emit the old LC_IPHONE_MIN_VERSION).
    let (major, minor) = ios_deployment_target();
    format!("{}-apple-ios{}.{}.0", arch.target_name(), major, minor)
}

fn ios_lld_platform_version() -> String {
    let (major, minor) = ios_deployment_target();
    format!("{major}.{minor}")
}

pub fn ios_sim_llvm_target(arch: Arch) -> String {
    let (major, minor) = ios_deployment_target();
    format!("{}-apple-ios{}.{}.0-simulator", arch.target_name(), major, minor)
}

fn tvos_deployment_target() -> (u32, u32) {
    // If you are looking for the default deployment target, prefer `rustc --print deployment-target`.
    from_set_deployment_target("TVOS_DEPLOYMENT_TARGET").unwrap_or((10, 0))
}

fn tvos_lld_platform_version() -> String {
    let (major, minor) = tvos_deployment_target();
    format!("{major}.{minor}")
}

pub fn tvos_llvm_target(arch: Arch) -> String {
    let (major, minor) = tvos_deployment_target();
    format!("{}-apple-tvos{}.{}.0", arch.target_name(), major, minor)
}

pub fn tvos_sim_llvm_target(arch: Arch) -> String {
    let (major, minor) = tvos_deployment_target();
    format!("{}-apple-tvos{}.{}.0-simulator", arch.target_name(), major, minor)
}

fn watchos_deployment_target() -> (u32, u32) {
    // If you are looking for the default deployment target, prefer `rustc --print deployment-target`.
    from_set_deployment_target("WATCHOS_DEPLOYMENT_TARGET").unwrap_or((5, 0))
}

fn watchos_lld_platform_version() -> String {
    let (major, minor) = watchos_deployment_target();
    format!("{major}.{minor}")
}

pub fn watchos_sim_llvm_target(arch: Arch) -> String {
    let (major, minor) = watchos_deployment_target();
    format!("{}-apple-watchos{}.{}.0-simulator", arch.target_name(), major, minor)
}
