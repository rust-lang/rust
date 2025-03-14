// This test check that we recognize all the well known config names
// and that we correctly lint on unexpected values.
//
// This test also serve as an "anti-regression" for the well known
// values since the suggestion shows them.
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()
//@ compile-flags: -Zcheck-cfg-all-expected

#![feature(cfg_overflow_checks)]
#![feature(cfg_relocation_model)]
#![feature(cfg_sanitize)]
#![feature(cfg_target_has_atomic)]
#![feature(cfg_target_has_atomic_equal_alignment)]
#![feature(cfg_target_thread_local)]
#![feature(cfg_ub_checks)]
#![feature(fmt_debug)]

// This part makes sure that none of the well known names are
// unexpected.
//
// BUT to make sure that no expected values changes without
// being noticed we pass them a obviously wrong value so the
// diagnostic prints the list of expected values.
#[cfg(any(
    // tidy-alphabetical-start
    clippy = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    debug_assertions = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    doc = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    doctest = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    fmt_debug = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    miri = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    overflow_checks = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    panic = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    proc_macro = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    relocation_model = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    rustfmt = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    sanitize = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_abi = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_arch = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_endian = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_env = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_family = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    // target_feature = "_UNEXPECTED_VALUE",
    // ^ tested in target_feature.rs
    target_has_atomic = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_has_atomic_equal_alignment = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_has_atomic_load_store = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_os = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_pointer_width = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_thread_local = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    target_vendor = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    ub_checks = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    unix = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    windows = "_UNEXPECTED_VALUE",
    //~^ WARN unexpected `cfg` condition value
    // tidy-alphabetical-end
))]
fn unexpected_values() {}

#[cfg(target_os = "linuz")] // testing that we suggest `linux`
//~^ WARNING unexpected `cfg` condition value
fn target_os_linux_misspell() {}

// The #[cfg]s below serve as a safeguard to make sure we
// don't lint when using an expected well-known name and
// value, only a small subset of all possible expected
// configs are tested, since we already test the names
// above and don't need to test all values, just different
// combinations (without value, with value, both...).

#[cfg(target_os = "linux")]
fn target_os_linux() {}

#[cfg(target_feature = "crt-static")] // pure rustc feature
fn target_feature() {}

#[cfg(target_has_atomic = "8")]
fn target_has_atomic_8() {}

#[cfg(target_has_atomic)]
fn target_has_atomic() {}

#[cfg(unix)]
fn unix() {}

#[cfg(doc)]
fn doc() {}

#[cfg(clippy)]
fn clippy() {}

#[cfg_attr(rustfmt, rustfmt::skip)]
fn rustfmt() {}

fn main() {}
