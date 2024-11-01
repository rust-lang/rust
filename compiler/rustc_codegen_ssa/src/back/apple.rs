use std::env;
use std::num::ParseIntError;

use rustc_session::Session;
use rustc_target::spec::Target;

#[cfg(test)]
mod tests;

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
fn deployment_target_env_var(os: &str) -> &'static str {
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

    if let Ok(deployment_target) = env::var(deployment_target_env_var(&sess.target.os)) {
        match parse_version(&deployment_target) {
            // It is common that the deployment target is set too low, e.g. on macOS Aarch64 to also
            // target older x86_64, the user may set a lower deployment target than supported.
            //
            // To avoid such issues, we silently raise the deployment target here.
            // FIXME: We want to show a warning when `version < os_min`.
            Ok(version) => version.max(min),
            // FIXME: Report erroneous environment variable to user.
            Err(_) => min,
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
