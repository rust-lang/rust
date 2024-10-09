use std::fmt::{Display, from_fn};
use std::io::ErrorKind;
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::{env, fs};

use rustc_session::Session;
use rustc_target::spec::Target;

use crate::errors::{AppleDeploymentTarget, AppleSdkError};

#[cfg(test)]
mod tests;

pub(super) fn sdk_name(target: &Target) -> &'static str {
    match (&*target.os, &*target.abi) {
        ("ios", "") => "iPhoneOS",
        ("ios", "sim") => "iPhoneSimulator",
        // Mac Catalyst uses the macOS SDK
        ("ios", "macabi") => "MacOSX",
        ("macos", "") => "MacOSX",
        ("tvos", "") => "AppleTVOS",
        ("tvos", "sim") => "AppleTVSimulator",
        ("visionos", "") => "XROS",
        ("visionos", "sim") => "XRSimulator",
        ("watchos", "") => "WatchOS",
        ("watchos", "sim") => "WatchSimulator",
        (os, abi) => unreachable!("invalid os '{os}' / abi '{abi}' combination for Apple target"),
    }
}

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

// TOCTOU is not _really_ an issue with our use of `try_exists` in here, we mostly use it for
// diagnostics, and these directories are global state that the user can change anytime anyhow in
// ways that are going to interfere much more with the compilation process.
fn try_exists(path: &Path) -> Result<bool, AppleSdkError> {
    path.try_exists().map_err(|error| AppleSdkError::FailedReading { path: path.to_owned(), error })
}

/// Get the SDK path for an SDK under `/Library/Developer/CommandLineTools`.
fn sdk_root_in_sdks_dir(sdks_dir: impl Into<PathBuf>, sdk_name: &str) -> PathBuf {
    let mut path = sdks_dir.into();
    path.push("SDKs");
    path.push(sdk_name);
    path.set_extension("sdk");
    path
}

/// Get the SDK path for an SDK under `/Applications/Xcode.app/Contents/Developer`.
fn sdk_root_in_developer_dir(developer_dir: impl Into<PathBuf>, sdk_name: &str) -> PathBuf {
    let mut path = developer_dir.into();
    path.push("Platforms");
    path.push(sdk_name);
    path.set_extension("platform");
    path.push("Developer");
    path.push("SDKs");
    path.push(sdk_name);
    path.set_extension("sdk");
    path
}

/// Find a SDK root from the user's environment for the given SDK name.
///
/// We do this by searching (purely by names in the filesystem, without reading SDKSettings.json)
/// for a matching SDK in the following places:
/// - `DEVELOPER_DIR`
/// - `/var/db/xcode_select_link`
/// - `/Applications/Xcode.app`
/// - `/Library/Developer/CommandLineTools`
///
/// This does roughly the same thing as `xcrun -sdk $sdk_name -show-sdk-path` (see `man xcrun` for
/// a few details on the search algorithm).
///
/// The reason why we implement this logic ourselves is:
/// - Reading these directly is faster than spawning a new process.
/// - `xcrun` can be fairly slow to start up after a reboot.
/// - In the future, we will be able to integrate this better with the compiler's change tracking
///   mechanisms, allowing rebuilds when the involved env vars and paths here change. See #118204.
/// - It's easier for us to emit better error messages.
///
/// Though a downside is that `xcrun` might be expanded in the future to check more places, and then
/// `rustc` would have to be changed to keep up. Furthermore, `xcrun`'s exact algorithm is
/// undocumented, so it might be doing more things than we do here.
pub(crate) fn find_sdk_root(sdk_name: &'static str) -> Result<PathBuf, AppleSdkError> {
    // Only try this if the host OS is macOS.
    if !cfg!(target_os = "macos") {
        return Err(AppleSdkError::MissingCrossCompileNonMacOS { sdk_name });
    }

    // NOTE: We could consider walking upwards in `SDKROOT` assuming Xcode directory structure, but
    // that isn't what `xcrun` does, and might still not yield the desired result (e.g. if using an
    // old SDK to compile for an old ARM iOS arch, we don't want `rustc` to pick a macOS SDK from
    // the old Xcode).

    // Try reading from `DEVELOPER_DIR`.
    if let Some(dir) = std::env::var_os("DEVELOPER_DIR") {
        let dir = PathBuf::from(dir);
        let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);

        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            let sdkroot_bare = sdk_root_in_sdks_dir(&dir, sdk_name);
            if try_exists(&sdkroot_bare)? {
                return Ok(sdkroot_bare);
            } else {
                return Err(AppleSdkError::MissingDeveloperDir { dir, sdkroot, sdkroot_bare });
            }
        }
    }

    // Next, try to read the link that `xcode-select` sets.
    //
    // FIXME(madsmtm): Support cases where `/var/db/xcode_select_link` contains a relative path?
    let path = Path::new("/var/db/xcode_select_link");
    match fs::read_link(path) {
        Ok(dir) => {
            let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);
            if try_exists(&sdkroot)? {
                return Ok(sdkroot);
            } else {
                let sdkroot_bare = sdk_root_in_sdks_dir(&dir, sdk_name);
                if try_exists(&sdkroot_bare)? {
                    return Ok(sdkroot_bare);
                } else {
                    return Err(AppleSdkError::MissingXcodeSelect { dir, sdkroot, sdkroot_bare });
                }
            }
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            // Intentionally ignore not found errors, if `xcode-select --reset` is called the
            // link will not exist.
        }
        Err(error) => return Err(AppleSdkError::FailedReading { path: path.into(), error }),
    }

    // Next, fall back to reading from `/Applications/Xcode.app`.
    let dir = Path::new("/Applications/Xcode.app/Contents/Developer");
    if try_exists(dir)? {
        let sdkroot = sdk_root_in_developer_dir(dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingXcode { sdkroot, sdk_name });
        }
    }

    // Finally, fall back to reading from `/Library/Developer/CommandLineTools`.
    let dir = Path::new("/Library/Developer/CommandLineTools");
    if try_exists(dir)? {
        let sdkroot = sdk_root_in_sdks_dir(dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingCommandlineTools { sdkroot, sdk_name });
        }
    }

    Err(AppleSdkError::Missing { sdk_name })
}
