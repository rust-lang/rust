use std::borrow::Cow;
use std::env;
use std::fs;
use std::io::ErrorKind;
use std::fmt::{Display, from_fn};
use std::path::{Path, PathBuf};

use rustc_session::Session;
use rustc_target::spec::{
    AppleOSVersion, apple_deployment_target_env_var, apple_minimum_deployment_target,
    apple_os_minimum_deployment_target, apple_parse_version,
};

use crate::errors::{AppleDeploymentTarget, AppleSdkError};

#[cfg(test)]
mod tests;

pub fn pretty_version(version: AppleOSVersion) -> impl Display {
    let (major, minor, patch) = version;
    from_fn(move |f| {
        write!(f, "{major}.{minor}")?;
        if patch != 0 {
            write!(f, ".{patch}")?;
        }
        Ok(())
    })
}

/// Get the deployment target based on the standard environment variables, or fall back to the
/// minimum version supported by `rustc`.
pub fn deployment_target(sess: &Session) -> AppleOSVersion {
    let os_min = apple_os_minimum_deployment_target(&sess.target.os);
    let min = apple_minimum_deployment_target(&sess.target);
    let env_var = apple_deployment_target_env_var(&sess.target.os);

    if let Ok(deployment_target) = env::var(env_var) {
        match apple_parse_version(&deployment_target) {
            Ok(version) => {
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

fn add_version_to_llvm_target(llvm_target: &str, deployment_target: AppleOSVersion) -> String {
    let mut components = llvm_target.split("-");
    let arch = components.next().expect("darwin target should have arch");
    let vendor = components.next().expect("darwin target should have vendor");
    let os = components.next().expect("darwin target should have os");
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

/// The target triple depends on the deployment target, and is required to
/// enable features such as cross-language LTO, and for picking the right
/// Mach-O commands.
///
/// Certain optimizations also depend on the deployment target.
pub fn versioned_llvm_target(sess: &Session) -> Cow<'static, str> {
    if sess.target.is_like_osx {
        add_version_to_llvm_target(&sess.target.llvm_target, deployment_target(sess)).into()
    } else {
        sess.target.llvm_target.clone()
    }
}

// TOCTOU is not _really_ an issue with our use of `try_exists` in here, we mostly use it for
// diagnostics, and these directories are global state that the user may change anytime anyhow.
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
    // Only try this if host OS is macOS.
    if !cfg!(target_os = "macos") {
        return Err(AppleSdkError::MissingCrossCompileNonMacOS { sdk_name });
    }

    // NOTE: We could consider walking upwards in `SDKROOT` assuming Xcode directory structure, but
    // that isn't what `xcrun` does, and might still not yield the desired result (e.g. if using an
    // old SDK to compile for an old ARM iOS arch, we don't want `rustc` to pick a macOS SDK from
    // the old Xcode).

    // Try reading from `DEVELOPER_DIR` on all hosts.
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
    let path = PathBuf::from("/var/db/xcode_select_link");
    match fs::read_link(&path) {
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
        Err(error) => return Err(AppleSdkError::FailedReading { path, error }),
    }

    // Next, fall back to reading from `/Applications/Xcode.app`.
    let dir = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
    if try_exists(&dir)? {
        let sdkroot = sdk_root_in_developer_dir(&dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingXcode { sdkroot });
        }
    }

    // Finally, fall back to reading from `/Library/Developer/CommandLineTools`.
    let dir = PathBuf::from("/Library/Developer/CommandLineTools");
    if try_exists(&dir)? {
        let sdkroot = sdk_root_in_sdks_dir(&dir, sdk_name);
        if try_exists(&sdkroot)? {
            return Ok(sdkroot);
        } else {
            return Err(AppleSdkError::MissingCommandlineTools { sdkroot });
        }
    }

    Err(AppleSdkError::Missing { sdk_name })
}
