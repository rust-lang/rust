use std::panic;

use crate::command::Command;
use crate::env_var;
use crate::util::handle_failed_output;

/// `TARGET`
#[must_use]
pub fn target() -> String {
    env_var("TARGET")
}

/// Check if target is windows-like.
#[must_use]
pub fn is_windows() -> bool {
    target().contains("windows")
}

/// Check if target uses msvc.
#[must_use]
pub fn is_msvc() -> bool {
    target().contains("msvc")
}

/// Check if target is windows-gnu.
#[must_use]
pub fn is_windows_gnu() -> bool {
    target().ends_with("windows-gnu")
}

/// Check if target is windows-msvc.
#[must_use]
pub fn is_windows_msvc() -> bool {
    target().ends_with("windows-msvc")
}

/// Check if target is win7.
#[must_use]
pub fn is_win7() -> bool {
    target().contains("win7")
}

/// Check if target uses macOS.
#[must_use]
pub fn is_darwin() -> bool {
    target().contains("darwin")
}

/// Check if target uses AIX.
#[must_use]
pub fn is_aix() -> bool {
    target().contains("aix")
}

/// Get the target OS on Apple operating systems.
#[must_use]
pub fn apple_os() -> &'static str {
    if target().contains("darwin") {
        "macos"
    } else if target().contains("ios") {
        "ios"
    } else if target().contains("tvos") {
        "tvos"
    } else if target().contains("watchos") {
        "watchos"
    } else if target().contains("visionos") {
        "visionos"
    } else {
        panic!("not an Apple OS")
    }
}

/// Check if `component` is within `LLVM_COMPONENTS`
#[must_use]
pub fn llvm_components_contain(component: &str) -> bool {
    // `LLVM_COMPONENTS` is a space-separated list of words
    env_var("LLVM_COMPONENTS").split_whitespace().find(|s| s == &component).is_some()
}

/// Run `uname`. This assumes that `uname` is available on the platform!
#[track_caller]
#[must_use]
pub fn uname() -> String {
    let caller = panic::Location::caller();
    let mut uname = Command::new("uname");
    let output = uname.run();
    if !output.status().success() {
        handle_failed_output(&uname, output, caller.line());
    }
    output.stdout_utf8()
}
