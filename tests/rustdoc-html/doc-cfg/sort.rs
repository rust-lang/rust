// Tests that `doc(cfg)` badges and auto-detected `cfg` badges are sorted deterministically.
//@ edition:2021
//@ compile-flags: --cfg target_os="linux"

#![crate_name = "foo"]
#![feature(doc_cfg)]

// TEST 1: Explicit `#[doc(cfg(...))]`
// Tests that OS targets are sorted alphabetically.
//@ has 'foo/fn.foo.html'
//@ has - '//*[@class="stab portability"]' 'Available on Android or Apple or Cygwin \
// or DragonFly BSD or FreeBSD or Linux or NetBSD or OpenBSD only.'
#[doc(cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_vendor = "apple",
    target_os = "cygwin"
)))]
pub fn foo() {}

// TEST 2: Implicit `#[cfg(...)]` via auto-detection
// Tests that targets are sorted alphabetically just like explicit `doc(cfg)`.
//@ has 'foo/fn.bar.html'
//@ has - '//*[@class="stab portability"]' 'Available on Android or Apple or Cygwin \
// or DragonFly BSD or FreeBSD or Linux or NetBSD or OpenBSD only.'
#[cfg(any(
    target_os = "android",
    target_os = "linux",
    target_os = "dragonfly",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_vendor = "apple",
    target_os = "cygwin"
))]
pub fn bar() {}
