// Tests that target-exclusive `doc(cfg)` badges are sorted by tier and alphabetically.
//@ edition:2021
#![crate_name = "foo"]
#![feature(doc_cfg)]

//@ has 'foo/fn.foo.html'
//@ has - '//*[@class="stab portability"]' 'Available on Android or Apple or Cygwin \
// or DragonFly BSD or FreeBSD or Linux or NetBSD or OpenBSD or QNX Neutrino only.'
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "nto",
    target_vendor = "apple",
    target_os = "cygwin"
))]
pub fn foo() {}
