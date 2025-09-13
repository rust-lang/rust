use std::borrow::Cow;
use std::fmt::{Display, from_fn};
use std::num::ParseIntError;
use std::str::FromStr;

use crate::spec::{
    BinaryFormat, Cc, DebuginfoKind, FloatAbi, FramePointer, LinkerFlavor, Lld, RustcAbi,
    SplitDebuginfo, StackProbeType, StaticCow, Target, TargetOptions, cvs,
};

#[cfg(test)]
mod tests;

use Arch::*;
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq)]
pub(crate) enum Arch {
    Armv7k,
    Armv7s,
    Arm64,
    Arm64e,
    Arm64_32,
    I386,
    I686,
    X86_64,
    X86_64h,
}

impl Arch {
    fn target_name(self) -> &'static str {
        match self {
            Armv7k => "armv7k",
            Armv7s => "armv7s",
            Arm64 => "arm64",
            Arm64e => "arm64e",
            Arm64_32 => "arm64_32",
            I386 => "i386",
            I686 => "i686",
            X86_64 => "x86_64",
            X86_64h => "x86_64h",
        }
    }

    pub(crate) fn target_arch(self) -> Cow<'static, str> {
        Cow::Borrowed(match self {
            Armv7k | Armv7s => "arm",
            Arm64 | Arm64e | Arm64_32 => "aarch64",
            I386 | I686 => "x86",
            X86_64 | X86_64h => "x86_64",
        })
    }

    fn target_cpu(self, env: TargetEnv) -> &'static str {
        match self {
            Armv7k => "cortex-a8",
            Armv7s => "swift", // iOS 10 is only supported on iPhone 5 or higher.
            Arm64 => match env {
                TargetEnv::Normal => "apple-a7",
                TargetEnv::Simulator => "apple-a12",
                TargetEnv::MacCatalyst => "apple-a12",
            },
            Arm64e => "apple-a12",
            Arm64_32 => "apple-s4",
            // Only macOS 10.12+ is supported, which means
            // all x86_64/x86 CPUs must be running at least penryn
            // https://github.com/llvm/llvm-project/blob/01f924d0e37a5deae51df0d77e10a15b63aa0c0f/clang/lib/Driver/ToolChains/Arch/X86.cpp#L79-L82
            I386 | I686 => "penryn",
            X86_64 => "penryn",
            // Note: `core-avx2` is slightly more advanced than `x86_64h`, see
            // comments (and disabled features) in `x86_64h_apple_darwin` for
            // details. It is a higher baseline then `penryn` however.
            X86_64h => "core-avx2",
        }
    }

    fn stack_probes(self) -> StackProbeType {
        match self {
            Armv7k | Armv7s => StackProbeType::None,
            Arm64 | Arm64e | Arm64_32 | I386 | I686 | X86_64 | X86_64h => StackProbeType::Inline,
        }
    }
}

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum TargetEnv {
    Normal,
    Simulator,
    MacCatalyst,
}

impl TargetEnv {
    fn target_env(self) -> &'static str {
        match self {
            Self::Normal => "",
            Self::MacCatalyst => "macabi",
            Self::Simulator => "sim",
        }
    }
}

/// Get the base target options, unversioned LLVM target and `target_arch` from the three
/// things that uniquely identify Rust's Apple targets: The OS, the architecture, and the ABI.
pub(crate) fn base(
    os: &'static str,
    arch: Arch,
    env: TargetEnv,
) -> (TargetOptions, StaticCow<str>, StaticCow<str>) {
    let mut opts = TargetOptions {
        llvm_floatabi: Some(FloatAbi::Hard),
        os: os.into(),
        env: env.target_env().into(),
        // NOTE: We originally set `cfg(target_abi = "macabi")` / `cfg(target_abi = "sim")`,
        // before it was discovered that those are actually environments:
        // https://github.com/rust-lang/rust/issues/133331
        //
        // But let's continue setting them for backwards compatibility.
        // FIXME(madsmtm): Warn about using these in the future.
        abi: env.target_env().into(),
        cpu: arch.target_cpu(env).into(),
        link_env_remove: link_env_remove(os),
        vendor: "apple".into(),
        linker_flavor: LinkerFlavor::Darwin(Cc::Yes, Lld::No),
        // macOS has -dead_strip, which doesn't rely on function_sections
        function_sections: false,
        dynamic_linking: true,
        families: cvs!["unix"],
        is_like_darwin: true,
        binary_format: BinaryFormat::MachO,
        // LLVM notes that macOS 10.11+ and iOS 9+ default
        // to v4, so we do the same.
        // https://github.com/llvm/llvm-project/blob/378778a0d10c2f8d5df8ceff81f95b6002984a4b/clang/lib/Driver/ToolChains/Darwin.cpp#L1203
        default_dwarf_version: 4,
        frame_pointer: match arch {
            // clang ignores `-fomit-frame-pointer` for Armv7, it only accepts `-momit-leaf-frame-pointer`
            Armv7k | Armv7s => FramePointer::Always,
            // clang supports omitting frame pointers for the rest, but... don't?
            Arm64 | Arm64e | Arm64_32 => FramePointer::NonLeaf,
            I386 | I686 | X86_64 | X86_64h => FramePointer::Always,
        },
        has_rpath: true,
        dll_suffix: ".dylib".into(),
        archive_format: "darwin".into(),
        // Thread locals became available with iOS 8 and macOS 10.7,
        // and both are far below our minimum.
        has_thread_local: true,
        abi_return_struct_as_int: true,
        emit_debug_gdb_scripts: false,
        eh_frame_header: false,
        stack_probes: arch.stack_probes(),

        debuginfo_kind: DebuginfoKind::DwarfDsym,
        // The historical default for macOS targets is to run `dsymutil` which
        // generates a packed version of debuginfo split from the main file.
        split_debuginfo: SplitDebuginfo::Packed,
        supported_split_debuginfo: Cow::Borrowed(&[
            SplitDebuginfo::Packed,
            SplitDebuginfo::Unpacked,
            SplitDebuginfo::Off,
        ]),

        // Tell the linker that we would like it to avoid irreproducible binaries.
        //
        // This environment variable is pretty magical but is intended for
        // producing deterministic builds. This was first discovered to be used
        // by the `ar` tool as a way to control whether or not mtime entries in
        // the archive headers were set to zero or not.
        //
        // In `ld64-351.8`, shipped with Xcode 9.3, the linker was updated to
        // read this flag too. Linker versions that don't support this flag
        // may embed modification timestamps in binaries (especially in debug
        // information).
        //
        // A cleaner alternative would be to pass the `-reproducible` flag,
        // though that is only supported since `ld64-819.6` shipped with Xcode
        // 14, which is too new for our minimum supported version:
        // https://doc.rust-lang.org/rustc/platform-support/apple-darwin.html#host-tooling
        //
        // For some more info see the commentary on #47086
        link_env: Cow::Borrowed(&[(Cow::Borrowed("ZERO_AR_DATE"), Cow::Borrowed("1"))]),

        ..Default::default()
    };
    if matches!(arch, Arch::I386 | Arch::I686) {
        // All Apple x86-32 targets have SSE2.
        opts.rustc_abi = Some(RustcAbi::X86Sse2);
    }
    (opts, unversioned_llvm_target(os, arch, env), arch.target_arch())
}

/// Generate part of the LLVM target triple.
///
/// See `rustc_codegen_ssa::back::versioned_llvm_target` for the full triple passed to LLVM and
/// Clang.
fn unversioned_llvm_target(os: &str, arch: Arch, env: TargetEnv) -> StaticCow<str> {
    let arch = arch.target_name();
    // Convert to the "canonical" OS name used by LLVM:
    // https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/TargetParser/Triple.cpp#L236-L282
    let os = match os {
        "macos" => "macosx",
        "ios" => "ios",
        "watchos" => "watchos",
        "tvos" => "tvos",
        "visionos" => "xros",
        _ => unreachable!("tried to get LLVM target OS for non-Apple platform"),
    };
    let environment = match env {
        TargetEnv::Normal => "",
        TargetEnv::MacCatalyst => "-macabi",
        TargetEnv::Simulator => "-simulator",
    };
    format!("{arch}-apple-{os}{environment}").into()
}

fn link_env_remove(os: &'static str) -> StaticCow<[StaticCow<str>]> {
    // Apple platforms only officially support macOS as a host for any compilation.
    //
    // If building for macOS, we go ahead and remove any erroneous environment state
    // that's only applicable to cross-OS compilation. Always leave anything for the
    // host OS alone though.
    if os == "macos" {
        // `IPHONEOS_DEPLOYMENT_TARGET` must not be set when using the Xcode linker at
        // "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/ld",
        // although this is apparently ignored when using the linker at "/usr/bin/ld".
        cvs!["IPHONEOS_DEPLOYMENT_TARGET", "TVOS_DEPLOYMENT_TARGET", "XROS_DEPLOYMENT_TARGET"]
    } else {
        // Otherwise if cross-compiling for a different OS/SDK (including Mac Catalyst), remove any part
        // of the linking environment that's wrong and reversed.
        cvs!["MACOSX_DEPLOYMENT_TARGET"]
    }
}

/// Deployment target or SDK version.
///
/// The size of the numbers in here are limited by Mach-O's `LC_BUILD_VERSION`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OSVersion {
    pub major: u16,
    pub minor: u8,
    pub patch: u8,
}

impl FromStr for OSVersion {
    type Err = ParseIntError;

    /// Parse an OS version triple (SDK version or deployment target).
    fn from_str(version: &str) -> Result<Self, ParseIntError> {
        if let Some((major, minor)) = version.split_once('.') {
            let major = major.parse()?;
            if let Some((minor, patch)) = minor.split_once('.') {
                Ok(Self { major, minor: minor.parse()?, patch: patch.parse()? })
            } else {
                Ok(Self { major, minor: minor.parse()?, patch: 0 })
            }
        } else {
            Ok(Self { major: version.parse()?, minor: 0, patch: 0 })
        }
    }
}

impl OSVersion {
    pub fn new(major: u16, minor: u8, patch: u8) -> Self {
        Self { major, minor, patch }
    }

    pub fn fmt_pretty(self) -> impl Display {
        let Self { major, minor, patch } = self;
        from_fn(move |f| {
            write!(f, "{major}.{minor}")?;
            if patch != 0 {
                write!(f, ".{patch}")?;
            }
            Ok(())
        })
    }

    pub fn fmt_full(self) -> impl Display {
        let Self { major, minor, patch } = self;
        from_fn(move |f| write!(f, "{major}.{minor}.{patch}"))
    }

    /// Minimum operating system versions currently supported by `rustc`.
    pub fn os_minimum_deployment_target(os: &str) -> Self {
        // When bumping a version in here, remember to update the platform-support docs too.
        //
        // NOTE: The defaults may change in future `rustc` versions, so if you are looking for the
        // default deployment target, prefer:
        // ```
        // $ rustc --print deployment-target
        // ```
        let (major, minor, patch) = match os {
            "macos" => (10, 12, 0),
            "ios" => (10, 0, 0),
            "tvos" => (10, 0, 0),
            "watchos" => (5, 0, 0),
            "visionos" => (1, 0, 0),
            _ => unreachable!("tried to get deployment target for non-Apple platform"),
        };
        Self { major, minor, patch }
    }

    /// The deployment target for the given target.
    ///
    /// This is similar to `os_minimum_deployment_target`, except that on certain targets it makes sense
    /// to raise the minimum OS version.
    ///
    /// This matches what LLVM does, see in part:
    /// <https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/TargetParser/Triple.cpp#L1900-L1932>
    pub fn minimum_deployment_target(target: &Target) -> Self {
        let (major, minor, patch) = match (&*target.os, &*target.arch, &*target.env) {
            ("macos", "aarch64", _) => (11, 0, 0),
            ("ios", "aarch64", "macabi") => (14, 0, 0),
            ("ios", "aarch64", "sim") => (14, 0, 0),
            ("ios", _, _) if target.llvm_target.starts_with("arm64e") => (14, 0, 0),
            // Mac Catalyst defaults to 13.1 in Clang.
            ("ios", _, "macabi") => (13, 1, 0),
            ("tvos", "aarch64", "sim") => (14, 0, 0),
            ("watchos", "aarch64", "sim") => (7, 0, 0),
            (os, _, _) => return Self::os_minimum_deployment_target(os),
        };
        Self { major, minor, patch }
    }
}

/// Name of the environment variable used to fetch the deployment target on the given OS.
pub fn deployment_target_env_var(os: &str) -> &'static str {
    match os {
        "macos" => "MACOSX_DEPLOYMENT_TARGET",
        "ios" => "IPHONEOS_DEPLOYMENT_TARGET",
        "watchos" => "WATCHOS_DEPLOYMENT_TARGET",
        "tvos" => "TVOS_DEPLOYMENT_TARGET",
        "visionos" => "XROS_DEPLOYMENT_TARGET",
        _ => unreachable!("tried to get deployment target env var for non-Apple platform"),
    }
}
