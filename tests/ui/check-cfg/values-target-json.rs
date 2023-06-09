// This test checks that we don't lint values defined by a custom target (target json)
//
// check-pass
// needs-llvm-components: x86
// compile-flags: --crate-type=lib --check-cfg=values() --target={{src-base}}/check-cfg/my-awesome-platform.json -Z unstable-options

#![feature(lang_items, no_core, auto_traits)]
#![no_core]

#[lang = "sized"]
trait Sized {}

#[cfg(target_os = "linuz")]
//~^ WARNING unexpected `cfg` condition value
fn target_os_linux_misspell() {}

#[cfg(target_os = "linux")]
fn target_os_linux() {}

#[cfg(target_os = "ericos")]
fn target_os_ericos() {}
