#![crate_name = "cfg_dep"]
#![crate_type = "rlib"]

#[cfg(target_os = "macos")]
#[cfg(rpass1)]
fn platform_private() -> &'static str {
    "macos v1"
}

#[cfg(target_os = "macos")]
#[cfg(any(rpass2, rpass3))]
fn platform_private() -> &'static str {
    "macos v2"
}

#[cfg(target_os = "linux")]
#[cfg(rpass1)]
fn platform_private() -> &'static str {
    "linux v1"
}

#[cfg(target_os = "linux")]
#[cfg(any(rpass2, rpass3))]
fn platform_private() -> &'static str {
    "linux v2"
}

#[cfg(target_os = "windows")]
#[cfg(rpass1)]
fn platform_private() -> &'static str {
    "windows v1"
}

#[cfg(target_os = "windows")]
#[cfg(any(rpass2, rpass3))]
fn platform_private() -> &'static str {
    "windows v2"
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
fn platform_private() -> &'static str {
    "other"
}

pub fn get_platform() -> &'static str {
    platform_private()
}

#[cfg(rpass3)]
fn _extra_private() {}
