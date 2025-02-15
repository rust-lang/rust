//! Test invalid syntax for `cfg(os_version_min)`.
#![feature(cfg_os_version_min)]

#[cfg(os_version_min = "macos")] //~ WARNING: unexpected `cfg` condition name: `os_version_min`
fn foo() {}

#[cfg(os_version_min("macos"))] //~ ERROR: expected two literals, a platform and a version
fn foo() {}
#[cfg(os_version_min("1", "2", "3"))] //~ ERROR: expected two literals, a platform and a version
fn foo() {}

#[cfg(os_version_min(macos, "1.0.0"))] //~ ERROR: expected a platform literal
fn foo() {}
#[cfg(os_version_min(42, "1.0.0"))] //~ ERROR: expected a platform literal
fn foo() {}

#[cfg(os_version_min("macos", 42))] //~ ERROR: expected a version literal
fn foo() {}
#[cfg(os_version_min("macos", 10.10))] //~ ERROR: expected a version literal
fn foo() {}
#[cfg(os_version_min("macos", false))] //~ ERROR: expected a version literal
fn foo() {}

#[cfg(os_version_min("11.0", "macos"))] //~ WARNING: unknown platform literal
fn foo() {}
#[cfg(os_version_min("linux", "5.3"))] //~ WARNING: unknown platform literal
fn foo() {}

#[cfg(os_version_min("windows", "10.0.10240"))] //~ ERROR: unimplemented platform
fn foo() {}

#[cfg(os_version_min("macos", "99999"))] //~ ERROR: failed parsing version
fn foo() {}
#[cfg(os_version_min("macos", "-1"))] //~ ERROR: failed parsing version
fn foo() {}
#[cfg(os_version_min("macos", "65536"))] //~ ERROR: failed parsing version
fn foo() {}
#[cfg(os_version_min("macos", "1.2.3.4"))] //~ ERROR: failed parsing version
fn foo() {}

#[cfg(os_version_min("macos", "10.0"))] //~ WARNING: version is set unnecessarily low
fn bar1() {}
#[cfg(os_version_min("macos", "0"))] //~ WARNING: version is set unnecessarily low
fn bar2() {}

#[cfg(os_version_min("macos", "10.12"))] //~ WARNING: version is set unnecessarily low
fn bar3() {}
#[cfg(os_version_min("ios", "10.0"))] //~ WARNING: version is set unnecessarily low
fn bar4() {}
#[cfg(os_version_min("tvos", "10.0"))] //~ WARNING: version is set unnecessarily low
fn bar5() {}
#[cfg(os_version_min("watchos", "5.0"))] //~ WARNING: version is set unnecessarily low
fn bar6() {}
#[cfg(os_version_min("visionos", "1.0"))] //~ WARNING: version is set unnecessarily low
fn bar7() {}

fn main() {}
