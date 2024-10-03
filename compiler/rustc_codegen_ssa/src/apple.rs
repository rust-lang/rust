use std::borrow::Cow;
use std::env;

use rustc_session::Session;
use rustc_target::spec::{
    AppleOSVersion, apple_deployment_target_env_var, apple_minimum_deployment_target,
    apple_parse_version,
};

#[cfg(test)]
mod tests;

/// Get the deployment target based on the standard environment variables, or fall back to the
/// minimum version supported by `rustc`.
pub fn deployment_target(sess: &Session) -> AppleOSVersion {
    let min = apple_minimum_deployment_target(&sess.target);

    if let Ok(deployment_target) = env::var(apple_deployment_target_env_var(&sess.target.os)) {
        match apple_parse_version(&deployment_target) {
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
