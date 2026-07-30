// Verifies that when compiling with -Zsanitizer=option,
// the `#[cfg(sanitize = "option")]` attribute is configured.

//@ add-minicore
//@ check-pass
//@ revisions: address cfi kcfi leak memory thread
//@compile-flags: -Ctarget-feature=-crt-static
//@[address]needs-sanitizer-address
//@[address]compile-flags: -Csanitizer=address -Zunstable-options
//@[cfi]needs-sanitizer-cfi
//@[cfi]compile-flags:     -Tsanitizer=cfi -Zunstable-options
//@[cfi]compile-flags:     -Clto -Ccodegen-units=1
//@[kcfi]needs-llvm-components: x86
//@[kcfi]compile-flags:    -Tsanitizer=kcfi --target x86_64-unknown-none -Zunstable-options
//@[kcfi]compile-flags:    -C panic=abort
//@[leak]needs-sanitizer-leak
//@[leak]compile-flags:    -Csanitizer=leak -Zunstable-options
//@[memory]needs-sanitizer-memory
//@[memory]compile-flags:  -Tsanitizer=memory -Zunstable-options
//@[thread]needs-sanitizer-thread
//@[thread]compile-flags:  -Tsanitizer=thread -Zunstable-options
//@ ignore-backends: gcc

#![feature(cfg_sanitize, no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[cfg(all(sanitize = "address", address))]
fn main() {}

#[cfg(all(sanitize = "cfi", cfi))]
fn main() {}

#[cfg(all(sanitize = "kcfi", kcfi))]
fn main() {}

#[cfg(all(sanitize = "leak", leak))]
fn main() {}

#[cfg(all(sanitize = "memory", memory))]
fn main() {}

#[cfg(all(sanitize = "thread", thread))]
fn main() {}

pub fn check() {
    main();
}
