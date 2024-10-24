use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::{env, fs};

use rustc_middle::bug;
use rustc_session::Session;
use rustc_session::config::CrateType;
use rustc_target::spec::{
    AppleOSVersion, Target, apple_deployment_target_env_var, apple_minimum_deployment_target,
    apple_os_minimum_deployment_target, apple_sdk_name,
};
use serde::Deserialize;

use crate::errors::{AppleDeploymentTarget, AppleSdkError};

#[cfg(test)]
mod tests;

/// Get the deployment target based on the standard environment variables, or fall back to the
/// minimum version supported by `rustc`.
pub fn deployment_target(sess: &Session) -> AppleOSVersion {
    let os_min = apple_os_minimum_deployment_target(&sess.target.os);
    let min = apple_minimum_deployment_target(&sess.target);
    let env_var = apple_deployment_target_env_var(&sess.target.os);

    if let Ok(deployment_target) = env::var(env_var) {
        match deployment_target.parse::<AppleOSVersion>() {
            Ok(version) => {
                // It is common that the deployment target is set a bit too low, for example on
                // macOS Aarch64 to also target older x86_64. So we only want to warn when variable
                // is lower than the minimum OS supported by rustc, not when the variable is lower
                // than the minimum for a specific target.
                if version < os_min {
                    sess.dcx().emit_warn(AppleDeploymentTarget::TooLow {
                        env_var,
                        version: version.pretty().to_string(),
                        os_min: os_min.pretty().to_string(),
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

    let AppleOSVersion { major, minor, patch } = deployment_target;

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
    // Platform directories are uninteresting to us, we only care about the inner SDK.
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
fn find_sdk_root(sdk_name: &'static str) -> Result<PathBuf, AppleSdkError> {
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

/// The architecture name understood by Apple's linker.
///
/// Supported architecture names can be found in the source:
/// https://github.com/apple-oss-distributions/ld64/blob/ld64-951.9/src/abstraction/MachOFileAbstraction.hpp#L578-L648
pub fn ld64_arch(target: &Target) -> &'static str {
    // `target.arch` / `target_arch` is not detailed enough.
    let llvm_arch = target.llvm_target.split_once('-').expect("LLVM target must have arch").0;

    // Intentially verbose to ensure that the list always matches correctly
    // with the list in the source above.
    match llvm_arch {
        "armv7k" => "armv7k",
        "armv7s" => "armv7s",
        "arm64" => "arm64",
        "arm64e" => "arm64e",
        "arm64_32" => "arm64_32",
        // ld64 doesn't understand i686, so fall back to i386 instead.
        //
        // Same story when linking with cc, since that ends up invoking ld64.
        "i386" | "i686" => "i386",
        "x86_64" => "x86_64",
        "x86_64h" => "x86_64h",
        _ => bug!("unsupported architecture {llvm_arch} in Apple target: {}", target.llvm_target),
    }
}

#[derive(Deserialize, Debug, Default)]
struct DefaultProperties {
    // Only set in macOS SDK.
    #[serde(rename = "IOS_UNZIPPERED_TWIN_PREFIX_PATH")]
    mac_catalyst_prefix_path: Option<PathBuf>,
}

#[derive(Deserialize, Debug)]
struct SupportedTargets {
    #[serde(rename = "Archs")]
    archs: BTreeSet<String>,

    #[serde(rename = "MaximumDeploymentTarget")]
    maximum_deployment_target: AppleOSVersion,
    // NOTE: We ignore `DefaultDeploymentTarget`, and let `rustc` choose the default instead. This
    // makes it easy when not going through Xcode to still support older OS versions.

    // NOTE: We ignore `MinimumDeploymentTarget`, since `rustc` generally has a very low deployment
    // target, and we don't want to warn in those common cases.
}

/// The parsed SDK information.
///
/// Note that the macOS SDK usually contains information related to Mac Catalyst as well, so if the
/// target is Mac Catalyst, more work is needed to extract the desired information.
///
/// `DefaultProperties` and `Variants` contain Xcode-specific information like the
/// `/System/iOSSupport` search paths (which we end up coding manually, it's not worth it to try
/// to parse).
///
/// FIXME(madsmtm): What does `IsBaseSDK` mean, and is it relevant for us?
#[derive(Deserialize, Debug)]
pub(crate) struct SDKSettings {
    #[serde(rename = "CanonicalName")]
    canonical_name: String,

    #[serde(rename = "Version")]
    version: AppleOSVersion,

    #[serde(rename = "MaximumDeploymentTarget")]
    maximum_deployment_target: AppleOSVersion,

    #[serde(rename = "DefaultProperties")]
    default_properties: DefaultProperties,

    /// Optional to support `SDKSettings.json` converted from an older `SDKSettings.plist`.
    #[serde(rename = "SupportedTargets")]
    supported_targets: Option<BTreeMap<String, SupportedTargets>>,

    /// Optional to support `SDKSettings.json` converted from an older `SDKSettings.plist`.
    ///
    /// Could in general be useful in the future for building "zippered" binaries, see:
    /// <https://github.com/rust-lang/rust/issues/131216>
    #[serde(rename = "VersionMap")]
    version_map: Option<BTreeMap<String, BTreeMap<AppleOSVersion, AppleOSVersion>>>,
}

impl SDKSettings {
    /// Attempt to parse required SDK settings from just the SDK's file name.
    ///
    /// This deliberately does not contain detailed error information, since it's a very rare case
    /// anyhow, so spending a lot of effort on nice error messages here is just wasted work.
    fn from_just_path(sdkroot: &Path) -> Option<Self> {
        let Some(extension) = sdkroot.extension() else {
            return None;
        };
        if extension != "sdk" {
            return None;
        }

        let file_stem: &str = sdkroot.file_stem()?.try_into().ok()?;

        // Strip e.g. "MacOSX14.0" to "14.0"
        let version = file_stem.trim_start_matches(|c: char| !c.is_ascii_digit());
        let version = version.parse().ok()?;

        Some(Self {
            // FIXME(madsmtm): Should we do verification of the canonical name too?
            canonical_name: file_stem.to_lowercase(),
            version,
            // The maximum supported version is often `major.minor.99`.
            maximum_deployment_target: AppleOSVersion {
                major: version.major,
                minor: version.minor,
                patch: u8::MAX,
            },
            default_properties: DefaultProperties::default(),
            supported_targets: None,
            version_map: None,
        })
    }

    /// Parse SDK settings from the given SDK root.
    fn from_sdkroot(sdkroot: &Path) -> Result<Self, AppleSdkError> {
        let path = sdkroot.join("SDKSettings.json");
        match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|error| {
                AppleSdkError::InvalidSDKSettingsJson { path, error: error.into() }
            }),
            // `SDKSettings.json` is present since macOS 10.14 and iOS 13.0 SDKs, so this must be an old SDK.
            Err(err) if err.kind() == ErrorKind::NotFound => {
                // Old SDKs must still have `SDKSettings.plist` though.
                if !try_exists(&sdkroot.join("SDKSettings.plist"))? {
                    return Err(AppleSdkError::MissingSDKSettings { sdkroot: sdkroot.to_owned() });
                }
                // We don't try to parse SDKSettings.plist though, since that'd require `rustc` to
                // contain a PList parser, which is a dependency we don't want.
                //
                // Clang doesn't do this either.

                // Try to extract SDK settings both from the standard path, and the canonicalized
                // path.
                let canonical_sdkroot = std::fs::canonicalize(sdkroot).map_err(|error| {
                    AppleSdkError::FailedReading { path: sdkroot.to_owned(), error }
                })?;
                Self::from_just_path(sdkroot)
                    .or_else(|| Self::from_just_path(&canonical_sdkroot))
                    .ok_or_else(|| AppleSdkError::NotSdkPath { sdkroot: sdkroot.to_owned() })
            }
            Err(error) => Err(AppleSdkError::FailedReading { path, error }),
        }
    }

    /// The value of settings["DefaultProperties"]["IOS_UNZIPPERED_TWIN_PREFIX_PATH"] or
    /// `/System/iOSSupport` if not set.
    ///
    /// This is only known to be `/System/iOSSupport`, but may change in the future.
    pub(crate) fn mac_catalyst_prefix_path(&self) -> &Path {
        self.default_properties
            .mac_catalyst_prefix_path
            .as_deref()
            .unwrap_or(Path::new("/System/iOSSupport"))
    }

    /// The version of the SDK.
    ///
    /// This is needed by the linker.
    ///
    /// settings["Version"] or settings["VersionMap"]["macOS_iOSMac"][settings["Version"]].
    pub(crate) fn sdk_version(&self, target: &Target, sdkroot: &Path) -> Result<AppleOSVersion, AppleSdkError> {
        if target.abi == "macabi" {
            let map = self
                .version_map
                .as_ref()
                .ok_or(AppleSdkError::SdkDoesNotSupportOS {
                    sdkroot: sdkroot.to_owned(),
                    os: target.os.clone(),
                    abi: target.abi.clone(),
                })?
                .get("macOS_iOSMac")
                .ok_or(AppleSdkError::SdkDoesNotSupportOS {
                    sdkroot: sdkroot.to_owned(),
                    os: target.os.clone(),
                    abi: target.abi.clone(),
                })?;
            map.get(&self.version).cloned().ok_or(AppleSdkError::MissingMacCatalystVersion {
                version: self.version.pretty().to_string(),
            })
        } else {
            Ok(self.version)
        }
    }

    fn supports_target(&self, target: &Target, sdkroot: &Path) -> Result<(), AppleSdkError> {
        let arch = ld64_arch(target);

        let sdk_name = apple_sdk_name(target).to_lowercase();
        let target_name = if target.abi == "macabi" { "iosmac" } else { &*sdk_name };

        if let Some(supported_targets) = &self.supported_targets {
            let Some(supported_target) = supported_targets.get(target_name) else {
                // If `settings["SupportedTargets"][sdk_name]` is not present
                return Err(AppleSdkError::SdkDoesNotSupportOS {
                    sdkroot: sdkroot.to_owned(),
                    os: target.os.clone(),
                    abi: target.abi.clone(),
                });
            };

            if !supported_target.archs.contains(arch) {
                // If the `Archs` key does not contain the expected architecture.
                return Err(AppleSdkError::SdkDoesNotSupportArch {
                    sdkroot: sdkroot.to_owned(),
                    arch,
                });
            }
        } else {
            // We try to guess if the SDK is compatible based on `CanonicalName`.
            if !self.canonical_name.contains(target_name) {
                return Err(AppleSdkError::SdkDoesNotSupportOS {
                    sdkroot: sdkroot.to_owned(),
                    os: target.os.clone(),
                    abi: target.abi.clone(),
                });
            }

            // Old SDKs without SupportedTargets do not have support for Aarch64 on macOS. This
            // check allows cross-compiling from Aarch64 to i686 by using the macOS 10.13 SDK.
            if target.os == "macos" && target.arch == "aarch64" {
                return Err(AppleSdkError::SdkDoesNotSupportArch {
                    sdkroot: sdkroot.to_owned(),
                    arch,
                });
            }
        }

        Ok(())
    }

    fn supports_deployment_target(
        &self,
        target: &Target,
        deployment_target: AppleOSVersion,
    ) -> Result<(), AppleDeploymentTarget> {
        let sdk_name = apple_sdk_name(target).to_lowercase();
        let target_name = if target.abi == "macabi" { "iosmac" } else { &*sdk_name };

        // settings["SupportedTargets"][target]["MaximumDeploymentTarget"] or
        // settings["MaximumDeploymentTarget"].
        let maximum_deployment_target = if let Some(supported_targets) = &self.supported_targets {
            if let Some(supported_target) = supported_targets.get(target_name) {
                supported_target.maximum_deployment_target
            } else {
                self.maximum_deployment_target
            }
        } else {
            self.maximum_deployment_target
        };

        if maximum_deployment_target < deployment_target {
            let env_var = apple_deployment_target_env_var(&target.os);
            return Err(AppleDeploymentTarget::TooHigh {
                sdk_max: maximum_deployment_target.pretty().to_string(),
                version: deployment_target.pretty().to_string(),
                env_var,
            });
        }

        Ok(())
    }

    fn try_get_sdkroot_environment(
        sess: &Session,
        crate_type: CrateType,
    ) -> Result<Option<(PathBuf, Self)>, AppleSdkError> {
        if let Some(sdkroot) = env::var_os("SDKROOT") {
            let sdkroot = PathBuf::from(&sdkroot);

            // Ignore `SDKROOT` if it's not a valid path. This is also what Clang does:
            // <https://github.com/llvm/llvm-project/blob/296a80102a9b72c3eda80558fb78a3ed8849b341/clang/lib/Driver/ToolChains/Darwin.cpp#L1661-L1678>
            if sdkroot == Path::new("/") {
                sess.dcx().emit_warn(AppleSdkError::SdkRootIsRootPath);
                return Ok(None);
            }
            if !sdkroot.is_absolute() {
                sess.dcx().emit_warn(AppleSdkError::SdkRootNotAbsolute { sdkroot });
                return Ok(None);
            }
            if !try_exists(&sdkroot)? {
                sess.dcx().emit_warn(AppleSdkError::SdkRootMissing { sdkroot });
                return Ok(None);
            }

            let settings = Self::from_sdkroot(&sdkroot)?;

            // Check if the SDK root is applicable for the current target, and ignore it if not.
            //
            // This can happen in many cases, including:
            // - When compiling proc-macros or build scripts in cross-compile scenarios with Cargo
            //   where `SDKROOT` is set for e.g. iOS.
            // - When running under bootstrap, which is invoked by the `/usr/bin/python3` binary;
            //   this is a trampoline that gets `SDKROOT` set to the macOS SDK default.
            // - Other invokers originating in trampoline binaries in `/usr/bin/*`.
            // - The user has set it and forgotten about it.
            if let Err(err) = settings.supports_target(&sess.target, &sdkroot) {
                // Best effort check to see if we're linking a build script.
                let is_build_script = matches!(&sess.opts.crate_name, Some(crate_name) if crate_name == "build_script_build");
                // Best effort check to see if we're linking a proc-macro
                let is_proc_macro = crate_type == CrateType::ProcMacro;

                // If we're likely to be building a build script or a proc macro.
                if sess.target.os == sess.host.os && (is_build_script || is_proc_macro) {
                    // Don't warn in this case.
                    return Ok(None);
                }

                // Report that we ignored the SDK.
                // FIXME(madsmtm): Make this diagnostic look nicer.
                sess.dcx().emit_warn(err);
                sess.dcx().emit_warn(AppleSdkError::SdkRootIgnored);

                return Ok(None);
            }

            Ok(Some((sdkroot, settings)))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn from_environment(
        sess: &Session,
        crate_type: CrateType,
    ) -> Result<(PathBuf, Self), AppleSdkError> {
        // Use `SDKROOT` if given and valid.
        let (sdkroot, settings) = if let Some((sdkroot, settings)) =
            Self::try_get_sdkroot_environment(sess, crate_type)?
        {
            (sdkroot, settings)
        } else {
            // Otherwise search for an SDK root.
            let sdkroot = find_sdk_root(apple_sdk_name(&sess.target))?;

            let settings = SDKSettings::from_sdkroot(&sdkroot)?;
            (sdkroot, settings)
        };

        // Finally, check that the found SDK matches what we expect.

        if let Err(err) = settings.supports_target(&sess.target, &sdkroot) {
            // Emit a warning if the SDK is not supported, but keep going with it (it may still
            // work even if unsupported, for example newer SDKs are marked as not supporting the
            // armv7/armv7s/armv7k architectures, but their `.tbd` files still support those).
            sess.dcx().emit_warn(err);
        }

        if let Err(err) = settings.supports_deployment_target(&sess.target, deployment_target(sess))
        {
            // Only warn here, the found SDK is probably still usable even if the deployment target
            // is not supported.
            sess.dcx().emit_warn(err);
        }

        Ok((sdkroot, settings))
    }
}
