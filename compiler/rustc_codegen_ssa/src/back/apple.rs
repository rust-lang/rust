use std::env;
use std::fmt::{Display, from_fn};
use std::num::ParseIntError;

use rustc_session::Session;
use rustc_target::spec::Target;

use crate::errors::AppleDeploymentTarget;

#[cfg(test)]
mod tests;

pub(super) fn macho_platform(target: &Target) -> u32 {
    match (&*target.os, &*target.abi) {
        ("macos", _) => object::macho::PLATFORM_MACOS,
        ("ios", "macabi") => object::macho::PLATFORM_MACCATALYST,
        ("ios", "sim") => object::macho::PLATFORM_IOSSIMULATOR,
        ("ios", _) => object::macho::PLATFORM_IOS,
        ("watchos", "sim") => object::macho::PLATFORM_WATCHOSSIMULATOR,
        ("watchos", _) => object::macho::PLATFORM_WATCHOS,
        ("tvos", "sim") => object::macho::PLATFORM_TVOSSIMULATOR,
        ("tvos", _) => object::macho::PLATFORM_TVOS,
        ("visionos", "sim") => object::macho::PLATFORM_XROSSIMULATOR,
        ("visionos", _) => object::macho::PLATFORM_XROS,
        _ => unreachable!("tried to get Mach-O platform for non-Apple target"),
    }
}

/// Add relocation and section data needed for a symbol to be considered
/// undefined by ld64.
///
/// Inherently very architecture-specific (unfortunately).
///
/// # New architectures
///
/// The values here can be found by compiling the following program:
///
/// ```c
/// void foo(void);
///
/// void use_foo() {
///     foo();
/// }
/// ```
///
/// With:
///
/// ```console
/// $ clang -c foo.c -O2 -g0 -fno-exceptions -target $CLANG_TARGET
/// ```
///
/// And then inspecting with `objdump -d foo.o` and/or:
///
/// ```no_run
/// use object::read::macho::{MachHeader, MachOFile};
/// use object::{File, Object, ObjectSection};
///
/// fn read(file: MachOFile<'_, impl MachHeader>) {
///     for section in file.sections() {
///         dbg!(section.name().unwrap(), section.data().unwrap(), section.align());
///         for reloc in section.relocations() {
///             dbg!(reloc);
///         }
///     }
/// }
///
/// fn main() {
///     match File::parse(&*std::fs::read("foo.o").unwrap()).unwrap() {
///         File::MachO32(file) => read(file),
///         File::MachO64(file) => read(file),
///         _ => unimplemented!(),
///     }
/// }
/// ```
pub(super) fn add_data_and_relocation(
    file: &mut object::write::Object<'_>,
    section: object::write::SectionId,
    symbol: object::write::SymbolId,
    target: &Target,
) -> object::write::Result<()> {
    let (data, align, addend, r_type): (&[u8], _, _, _) = match &*target.arch {
        "arm" => (&[0xff, 0xf7, 0xfe, 0xbf], 2, 0, object::macho::ARM_THUMB_RELOC_BR22),
        "aarch64" => (&[0, 0, 0, 0x14], 4, 0, object::macho::ARM64_RELOC_BRANCH26),
        "x86_64" => (
            &[0x55, 0x48, 0x89, 0xe5, 0x5d, 0xe9, 0, 0, 0, 0],
            16,
            -4,
            object::macho::X86_64_RELOC_BRANCH,
        ),
        "x86" => (&[0x55, 0x89, 0xe5, 0x5d, 0xe9, 0xf7, 0xff, 0xff, 0xff], 16, -4, 0),
        arch => unimplemented!("unsupported Apple architecture {arch:?}"),
    };

    let offset = file.section_mut(section).append_data(data, align);
    file.add_relocation(section, object::write::Relocation {
        offset,
        addend,
        symbol,
        flags: object::write::RelocationFlags::MachO { r_type, r_pcrel: true, r_length: 2 },
    })?;

    Ok(())
}

/// Deployment target or SDK version.
///
/// The size of the numbers in here are limited by Mach-O's `LC_BUILD_VERSION`.
type OSVersion = (u16, u8, u8);

/// Parse an OS version triple (SDK version or deployment target).
fn parse_version(version: &str) -> Result<OSVersion, ParseIntError> {
    if let Some((major, minor)) = version.split_once('.') {
        let major = major.parse()?;
        if let Some((minor, patch)) = minor.split_once('.') {
            Ok((major, minor.parse()?, patch.parse()?))
        } else {
            Ok((major, minor.parse()?, 0))
        }
    } else {
        Ok((version.parse()?, 0, 0))
    }
}

pub fn pretty_version(version: OSVersion) -> impl Display {
    let (major, minor, patch) = version;
    from_fn(move |f| {
        write!(f, "{major}.{minor}")?;
        if patch != 0 {
            write!(f, ".{patch}")?;
        }
        Ok(())
    })
}

/// Minimum operating system versions currently supported by `rustc`.
fn os_minimum_deployment_target(os: &str) -> OSVersion {
    // When bumping a version in here, remember to update the platform-support docs too.
    //
    // NOTE: The defaults may change in future `rustc` versions, so if you are looking for the
    // default deployment target, prefer:
    // ```
    // $ rustc --print deployment-target
    // ```
    match os {
        "macos" => (10, 12, 0),
        "ios" => (10, 0, 0),
        "tvos" => (10, 0, 0),
        "watchos" => (5, 0, 0),
        "visionos" => (1, 0, 0),
        _ => unreachable!("tried to get deployment target for non-Apple platform"),
    }
}

/// The deployment target for the given target.
///
/// This is similar to `os_minimum_deployment_target`, except that on certain targets it makes sense
/// to raise the minimum OS version.
///
/// This matches what LLVM does, see in part:
/// <https://github.com/llvm/llvm-project/blob/llvmorg-18.1.8/llvm/lib/TargetParser/Triple.cpp#L1900-L1932>
fn minimum_deployment_target(target: &Target) -> OSVersion {
    match (&*target.os, &*target.arch, &*target.abi) {
        ("macos", "aarch64", _) => (11, 0, 0),
        ("ios", "aarch64", "macabi") => (14, 0, 0),
        ("ios", "aarch64", "sim") => (14, 0, 0),
        ("ios", _, _) if target.llvm_target.starts_with("arm64e") => (14, 0, 0),
        // Mac Catalyst defaults to 13.1 in Clang.
        ("ios", _, "macabi") => (13, 1, 0),
        ("tvos", "aarch64", "sim") => (14, 0, 0),
        ("watchos", "aarch64", "sim") => (7, 0, 0),
        (os, _, _) => os_minimum_deployment_target(os),
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

/// Get the deployment target based on the standard environment variables, or fall back to the
/// minimum version supported by `rustc`.
pub fn deployment_target(sess: &Session) -> OSVersion {
    let min = minimum_deployment_target(&sess.target);
    let env_var = deployment_target_env_var(&sess.target.os);

    if let Ok(deployment_target) = env::var(env_var) {
        match parse_version(&deployment_target) {
            Ok(version) => {
                let os_min = os_minimum_deployment_target(&sess.target.os);
                // It is common that the deployment target is set a bit too low, for example on
                // macOS Aarch64 to also target older x86_64. So we only want to warn when variable
                // is lower than the minimum OS supported by rustc, not when the variable is lower
                // than the minimum for a specific target.
                if version < os_min {
                    sess.dcx().emit_warn(AppleDeploymentTarget::TooLow {
                        env_var,
                        version: pretty_version(version).to_string(),
                        os_min: pretty_version(os_min).to_string(),
                    });
                }

                // Raise the deployment target to the minimum supported.
                version.max(min)
            }
            Err(error) => {
                sess.dcx().emit_err(AppleDeploymentTarget::Invalid { env_var, error });
                min
            }
        }
    } else {
        // If no deployment target variable is set, default to the minimum found above.
        min
    }
}

pub(super) fn add_version_to_llvm_target(
    llvm_target: &str,
    deployment_target: OSVersion,
) -> String {
    let mut components = llvm_target.split("-");
    let arch = components.next().expect("apple target should have arch");
    let vendor = components.next().expect("apple target should have vendor");
    let os = components.next().expect("apple target should have os");
    let environment = components.next();
    assert_eq!(components.next(), None, "too many LLVM triple components");

    let (major, minor, patch) = deployment_target;

    assert!(
        !os.contains(|c: char| c.is_ascii_digit()),
        "LLVM target must not already be versioned"
    );

    if let Some(env) = environment {
        // Insert version into OS, before environment
        format!("{arch}-{vendor}-{os}{major}.{minor}.{patch}-{env}")
    } else {
        format!("{arch}-{vendor}-{os}{major}.{minor}.{patch}")
    }
}
